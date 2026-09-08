from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..config import ContactSheetConfig
from ..models import Candidate, SampleFrame


@dataclass
class VisualCard:
    item_id: str
    image_path: Path
    line1: str
    line2: str = ""


class ContactSheetBuilder:
    def __init__(self, config: ContactSheetConfig) -> None:
        self.config = config
        self.font = ImageFont.load_default(size=18)
        self.small_font = ImageFont.load_default(size=15)

    @staticmethod
    def candidate_card(candidate: Candidate) -> VisualCard:
        if candidate.image_path is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no image")
        time_label = "?" if candidate.pts_time is None else f"{candidate.pts_time:.2f}s"
        frame_label = "?" if candidate.frame_idx is None else str(candidate.frame_idx)
        return VisualCard(
            item_id=candidate.candidate_id,
            image_path=candidate.image_path,
            line1=f"{candidate.candidate_id} | frame {frame_label}",
            line2=f"{time_label} | retrieval {candidate.retrieval_score:.4f}",
        )

    @staticmethod
    def sample_card(sample: SampleFrame) -> VisualCard:
        return VisualCard(
            item_id=sample.sample_id,
            image_path=sample.image_path,
            line1=f"{sample.sample_id} | {sample.video_id}",
            line2=f"source time {sample.pts_time:.2f}s",
        )

    def _render_card(self, card: VisualCard) -> Image.Image:
        cfg = self.config
        canvas = Image.new("RGB", (cfg.card_width, cfg.card_height), "white")
        with Image.open(card.image_path) as source:
            image = ImageOps.contain(
                source.convert("RGB"),
                (cfg.card_width - 8, cfg.image_height),
                method=Image.Resampling.LANCZOS,
            )
        left = (cfg.card_width - image.width) // 2
        canvas.paste(image, (left, 4))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, cfg.card_width - 1, cfg.card_height - 1), outline="#555555")
        draw.rectangle(
            (2, cfg.image_height + 6, cfg.card_width - 3, cfg.image_height + 34),
            fill="#111111",
        )
        draw.text((8, cfg.image_height + 9), card.line1, fill="white", font=self.font)
        draw.text(
            (8, cfg.image_height + 40), card.line2, fill="#111111", font=self.small_font
        )
        return canvas

    def build(self, cards: list[VisualCard], output_dir: Path, prefix: str) -> list[Path]:
        if not cards:
            raise ValueError("Cannot build an empty contact sheet")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        per_sheet = self.config.cards_per_sheet
        for page, offset in enumerate(range(0, len(cards), per_sheet), start=1):
            chunk = cards[offset : offset + per_sheet]
            columns = min(self.config.columns, len(chunk))
            rows = math.ceil(len(chunk) / columns)
            sheet = Image.new(
                "RGB",
                (columns * self.config.card_width, rows * self.config.card_height),
                "#d8d8d8",
            )
            for index, card in enumerate(chunk):
                rendered = self._render_card(card)
                x = (index % columns) * self.config.card_width
                y = (index // columns) * self.config.card_height
                sheet.paste(rendered, (x, y))
            path = output_dir / f"{prefix}_{page:02d}.jpg"
            sheet.save(path, quality=self.config.jpeg_quality, subsampling=0)
            paths.append(path)
        return paths

