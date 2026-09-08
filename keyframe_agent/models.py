from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Criterion:
    criterion_id: str
    criterion_type: str
    description: str
    value: str = ""
    required: bool = True
    classification_confidence: float = 1.0
    reason: str = ""
    evidence_scope: str = "frame"


@dataclass
class EventPlan:
    event_id: str
    description_vi: str
    visual_query_en: str
    criteria: list[Criterion] = field(default_factory=list)


@dataclass
class QueryPlan:
    original_query: str
    main_visual_query_en: str
    support_queries_en: list[str] = field(default_factory=list)
    must_have: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    needs_motion: bool = False
    needs_text_reading: bool = False
    search_notes: str = ""
    fallback_visual_query_en: str = ""
    criteria: list[Criterion] = field(default_factory=list)
    events: list[EventPlan] = field(default_factory=list)
    sequence_required: bool = False
    task_type: str = "kis"
    qa_question: str = ""
    target_object: str = ""
    answer_expected_visible: bool = False
    # Planner-selected internal QA route. pipeline.run(..., qa_mode=...) may
    # override this without changing task_type="qa".
    qa_mode: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    candidate_id: str
    video_id: str
    ordinal: int
    retrieval_score: float
    main_score: float
    support_scores: list[float] = field(default_factory=list)
    frame_idx: int | None = None
    pts_time: float | None = None
    fps: float | None = None
    image_path: Path | None = None
    source_round: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["keyframe_ordinal"] = self.ordinal
        if self.image_path is not None:
            data["image_path"] = str(self.image_path)
        return data


@dataclass
class SampleFrame:
    sample_id: str
    video_id: str
    pts_time: float
    image_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "pts_time": self.pts_time,
            "image_path": str(self.image_path),
        }
