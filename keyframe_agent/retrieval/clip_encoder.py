from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPTextEncoder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        clean = [text.strip() for text in texts if text and text.strip()]
        if not clean:
            raise ValueError("At least one retrieval query is required")
        tokens = self.processor(
            text=clean,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.inference_mode():
            features = self.model.get_text_features(**tokens)
        # transformers versions differ: older releases return a Tensor while
        # newer releases may wrap it in BaseModelOutputWithPooling.
        if not torch.is_tensor(features):
            pooled = getattr(features, "pooler_output", None)
            if pooled is None:
                pooled = getattr(features, "text_embeds", None)
            if pooled is None:
                raise TypeError(
                    f"Unsupported CLIP text output type: {type(features).__name__}"
                )
            features = pooled
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().cpu().numpy().astype(np.float32)
