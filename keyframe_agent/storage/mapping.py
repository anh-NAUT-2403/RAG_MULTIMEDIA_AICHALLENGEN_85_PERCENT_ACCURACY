from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

from ..models import Candidate


class KeyframeMapping:
    def __init__(self, mapping_zip: Path) -> None:
        self.mapping_zip = Path(mapping_zip)
        self._cache: dict[str, list[dict[str, str]]] = {}

    def rows(self, video_id: str) -> list[dict[str, str]]:
        if video_id not in self._cache:
            member = f"map-keyframes/{video_id}.csv"
            with zipfile.ZipFile(self.mapping_zip) as archive:
                with archive.open(member) as stream:
                    reader = csv.DictReader(
                        io.TextIOWrapper(stream, encoding="utf-8-sig")
                    )
                    rows = list(reader)
            rows.sort(key=lambda row: int(row.get("n", 0)))
            self._cache[video_id] = rows
        return self._cache[video_id]

    def enrich(self, candidate: Candidate) -> Candidate:
        rows = self.rows(candidate.video_id)
        index = candidate.ordinal - 1
        if index < 0 or index >= len(rows):
            raise IndexError(
                f"Ordinal {candidate.ordinal} is absent for {candidate.video_id}"
            )
        row = rows[index]
        candidate.frame_idx = int(row.get("frame_idx", row.get("frame_id", "0")))
        candidate.pts_time = float(row["pts_time"])
        candidate.fps = float(row.get("fps", 0.0) or 0.0)
        return candidate

    def candidate_at(self, video_id: str, ordinal: int) -> Candidate:
        candidate = Candidate(
            candidate_id=f"{video_id}__K{ordinal:03d}",
            video_id=video_id,
            ordinal=ordinal,
            retrieval_score=0.0,
            main_score=0.0,
        )
        return self.enrich(candidate)

    def nearest_time(self, video_id: str, pts_time: float) -> Candidate:
        rows = self.rows(video_id)
        if not rows:
            raise ValueError(f"No keyframe mapping for {video_id}")
        best_index = min(
            range(len(rows)), key=lambda index: abs(float(rows[index]["pts_time"]) - pts_time)
        )
        return self.candidate_at(video_id, best_index + 1)

