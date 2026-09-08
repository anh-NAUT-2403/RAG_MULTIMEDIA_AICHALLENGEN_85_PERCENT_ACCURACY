from __future__ import annotations

from pathlib import Path

from ..models import Candidate
from .mapping import KeyframeMapping
from .zip_catalog import ZipCatalog


class SelectiveAssetStore:
    def __init__(
        self,
        data_root: Path,
        keyframe_zip_glob: str,
        video_zip_glob: str,
        cache_dir: Path,
        mapping: KeyframeMapping,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.mapping = mapping
        keyframe_zips = list(Path(data_root).glob(keyframe_zip_glob))
        video_zips = list(Path(data_root).glob(video_zip_glob))
        if not keyframe_zips:
            raise FileNotFoundError(f"No keyframe ZIP matches {keyframe_zip_glob}")
        self.keyframes = ZipCatalog(keyframe_zips)
        self.videos = ZipCatalog(video_zips)

    def extract_keyframe(self, candidate: Candidate) -> Path:
        if candidate.frame_idx is None:
            self.mapping.enrich(candidate)
        member = f"keyframes/{candidate.video_id}/{candidate.ordinal:03d}.jpg"
        output = (
            self.cache_dir
            / "keyframes"
            / candidate.video_id
            / f"{candidate.ordinal:03d}.jpg"
        )
        candidate.image_path = self.keyframes.extract(member, output)
        return candidate.image_path

    def neighbors(self, candidate: Candidate, radius: int) -> list[Candidate]:
        row_count = len(self.mapping.rows(candidate.video_id))
        start = max(1, candidate.ordinal - radius)
        end = min(row_count, candidate.ordinal + radius)
        results: list[Candidate] = []
        for ordinal in range(start, end + 1):
            item = self.mapping.candidate_at(candidate.video_id, ordinal)
            item.retrieval_score = candidate.retrieval_score
            item.main_score = candidate.main_score
            item.support_scores = candidate.support_scores
            item.source_round = candidate.source_round
            self.extract_keyframe(item)
            results.append(item)
        return results

    def extract_video(self, video_id: str) -> Path:
        if not self.videos.zip_paths:
            raise FileNotFoundError("No video ZIP files were configured")
        output = self.cache_dir / "videos" / f"{video_id}.mp4"
        return self.videos.extract_suffix(f"/{video_id}.mp4", output)

