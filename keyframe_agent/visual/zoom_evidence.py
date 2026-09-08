from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..models import Candidate
from .contact_sheet import ContactSheetBuilder, VisualCard


class ZoomEvidenceBuilder:
    """Creates detail crops that remain explicitly tied to their source frame."""

    _REGION_STARTS = {"top": 0.0, "center": 0.5, "bottom": 1.0}

    def __init__(self, detail_regions: list[str]) -> None:
        self.detail_regions = detail_regions

    def build(
        self,
        candidates: list[Candidate],
        sheets: ContactSheetBuilder,
        output_dir: Path,
        include_detail_crops: bool,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_cards = [sheets.candidate_card(candidate) for candidate in candidates]
        paths = sheets.build(full_cards, output_dir, "zoom_full")
        if not include_detail_crops or not self.detail_regions:
            return paths

        crop_dir = output_dir / "detail_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        detail_cards: list[VisualCard] = []
        for candidate in candidates:
            if candidate.image_path is None:
                continue
            with Image.open(candidate.image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            crop_height = max(1, round(height * 0.5))
            for region in self.detail_regions:
                start = round((height - crop_height) * self._REGION_STARTS[region])
                crop = image.crop((0, start, width, start + crop_height))
                crop_path = crop_dir / f"{candidate.candidate_id}_{region}.jpg"
                crop.save(crop_path, quality=95, subsampling=0)
                detail_cards.append(
                    VisualCard(
                        item_id=f"{candidate.candidate_id} | {region} detail",
                        image_path=crop_path,
                        line1=f"{candidate.candidate_id} | {region} detail",
                        line2="Crop from the same candidate frame.",
                    )
                )
        if detail_cards:
            paths.extend(sheets.build(detail_cards, output_dir, "zoom_detail"))
        return paths
