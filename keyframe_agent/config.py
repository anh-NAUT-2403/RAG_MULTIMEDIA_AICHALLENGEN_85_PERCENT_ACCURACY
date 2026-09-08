from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: Path
    feature_zip: Path
    mapping_zip: Path
    keyframe_zip_glob: str
    video_zip_glob: str
    cache_dir: Path
    output_dir: Path


@dataclass
class RetrievalConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    main_weight: float = 0.70
    support_weight: float = 0.15
    max_support_queries: int = 2
    top_per_video: int = 4
    shortlist_size: int = 40
    max_candidates_per_video: int = 3
    max_rounds: int = 2
    visual_accept_score: int = 72


@dataclass
class ContactSheetConfig:
    columns: int = 4
    cards_per_sheet: int = 20
    card_width: int = 420
    card_height: int = 286
    image_height: int = 224
    jpeg_quality: int = 92


@dataclass
class OpenAIConfig:
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-5.4-mini"
    reasoning_effort: str = "low"
    image_detail: str = "high"
    max_output_tokens: int = 7000
    retries: int = 3
    retry_base_seconds: float = 2.0
    # Number of neighboring frames the final VLM ranks for human comparison.
    # Keep this between 5 and 10. The first verified row remains the main answer.
    final_candidate_count: int = 8


@dataclass
class RefinementConfig:
    finalists: int = 5
    neighbor_radius: int = 3
    use_raw_video_for_motion: bool = True
    raw_video_finalists: int = 2
    video_window_before: float = 5.0
    video_window_after: float = 5.0
    static_fps: float = 1.0
    motion_fps: float = 3.0
    max_sampled_frames: int = 30
    final_confidence_threshold: float = 0.55


@dataclass
class VerificationConfig:
    """Generic hard gates derived from query criteria, never query-specific words."""

    enabled: bool = True
    hard_gate_types: list[str] = field(
        default_factory=lambda: [
            "visual",
            "exact_text",
            "text_presence",
            "count",
            "temporal",
        ]
    )
    classification_confidence_threshold: float = 0.65
    criterion_confidence_threshold: float = 0.65


@dataclass
class MultiEventConfig:
    """Retrieval and verification limits for ordered E1/E2/... TRAKE queries."""

    enabled: bool = True
    candidate_videos: int = 3
    top_frames_per_event_per_video: int = 4
    neighbor_radius: int = 2
    max_candidates_per_video: int = 80
    uniform_fallback_enabled: bool = True
    uniform_fallback_videos: int = 1
    uniform_fallback_frames: int = 48
    uniform_fallback_max_candidates: int = 120


@dataclass
class QAConfig:
    """Bounded, video-level evidence search for visual question answering."""

    enabled: bool = True
    result_count: int = 5
    retrieval_shortlist_size: int = 120
    candidate_videos: int = 5
    max_rounds: int = 2
    fallback_on_unverified: bool = True
    screening_enabled: bool = True
    screening_videos: int = 30
    screening_frames_per_video: int = 8
    max_retrieval_criteria: int = 7
    top_frames_per_criterion_per_video: int = 2
    coverage_rank_cutoff: int = 80
    coverage_weight: float = 0.50
    evidence_score_weight: float = 0.30
    main_query_weight: float = 0.20
    representatives_per_video: int = 24
    max_candidates_per_video: int = 32
    minimum_confidence: float = 0.75


@dataclass
class QALongRangeConfig:
    """Extra bounded work for QA whose evidence is far apart in one video."""

    enabled: bool = True
    auto_detect: bool = True
    preserve_main_query_videos: int = 3
    include_event_queries: bool = True
    max_event_queries: int = 4
    answer_locator_query_enabled: bool = True
    screening_frames_per_video: int = 12
    representatives_per_video: int = 48
    max_candidates_per_video: int = 64


@dataclass
class TemporalKISConfig:
    """Video-grounded verification for KIS descriptions spanning several moments."""

    enabled: bool = True
    candidate_videos: int = 5
    max_rounds: int = 2
    fallback_on_unverified: bool = True
    retrieval_shortlist_size: int = 120
    screening_enabled: bool = True
    screening_videos: int = 30
    screening_frames_per_video: int = 8
    max_retrieval_criteria: int = 7
    top_frames_per_criterion_per_video: int = 2
    coverage_rank_cutoff: int = 80
    coverage_weight: float = 0.50
    evidence_score_weight: float = 0.30
    main_query_weight: float = 0.20
    representatives_per_video: int = 24
    max_candidates_per_video: int = 32
    minimum_confidence: float = 0.65


@dataclass
class RescueZoomConfig:
    """One bounded, high-detail verification pass for strong primary evidence."""

    enabled: bool = True
    min_primary_score: int = 75
    preserve_primary_videos: int = 2
    max_candidates: int = 16
    neighbor_radius: int = 4
    columns: int = 2
    cards_per_sheet: int = 8
    card_width: int = 700
    card_height: int = 540
    image_height: int = 420
    crop_text_when_needed: bool = True
    text_detail_regions: list[str] = field(default_factory=lambda: ["bottom"])


@dataclass
class FallbackConfig:
    """Recall-oriented recovery path after primary and rescue verification fail."""

    enabled: bool = True
    trigger_on_unverified: bool = True
    relaxed_query_enabled: bool = True
    relaxed_shortlist_size: int = 180
    candidate_videos: int = 12
    representative_frames_per_video: int = 16
    selected_videos_for_deep_scan: int = 2
    preserve_primary_videos: int = 1
    local_scan_anchors_per_video: int = 2
    local_scan_radius: int = 15
    deep_scan_batch_size: int = 40
    deep_scan_keep_per_video: int = 2


@dataclass
class RuntimeConfig:
    keep_extracted_videos: bool = False
    retrieval_only: bool = False
    task_type_override: str = "auto"
    qa_mode_override: str = "auto"


@dataclass
class AppConfig:
    data: DataConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    contact_sheet: ContactSheetConfig = field(default_factory=ContactSheetConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    multi_event: MultiEventConfig = field(default_factory=MultiEventConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    qa_long_range: QALongRangeConfig = field(default_factory=QALongRangeConfig)
    temporal_kis: TemporalKISConfig = field(default_factory=TemporalKISConfig)
    rescue_zoom: RescueZoomConfig = field(default_factory=RescueZoomConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.data.feature_zip.is_file():
            raise FileNotFoundError(f"Feature ZIP not found: {self.data.feature_zip}")
        if not self.data.mapping_zip.is_file():
            raise FileNotFoundError(f"Mapping ZIP not found: {self.data.mapping_zip}")
        if self.retrieval.shortlist_size < 1:
            raise ValueError("retrieval.shortlist_size must be positive")
        if not 0.0 < self.retrieval.main_weight <= 1.0:
            raise ValueError("retrieval.main_weight must be in (0, 1]")
        if self.contact_sheet.cards_per_sheet < 1:
            raise ValueError("contact_sheet.cards_per_sheet must be positive")
        if not 5 <= self.openai.final_candidate_count <= 10:
            raise ValueError("openai.final_candidate_count must be between 5 and 10")
        allowed_criterion_types = {
            "visual",
            "exact_text",
            "text_presence",
            "count",
            "temporal",
            "context",
        }
        if not set(self.verification.hard_gate_types).issubset(allowed_criterion_types):
            raise ValueError("verification.hard_gate_types contains an unknown type")
        if not 0 <= self.verification.classification_confidence_threshold <= 1:
            raise ValueError(
                "verification.classification_confidence_threshold must be in [0, 1]"
            )
        if not 0 <= self.verification.criterion_confidence_threshold <= 1:
            raise ValueError(
                "verification.criterion_confidence_threshold must be in [0, 1]"
            )
        if self.multi_event.candidate_videos < 1:
            raise ValueError("multi_event.candidate_videos must be positive")
        if self.multi_event.top_frames_per_event_per_video < 1:
            raise ValueError(
                "multi_event.top_frames_per_event_per_video must be positive"
            )
        if self.multi_event.neighbor_radius < 0:
            raise ValueError("multi_event.neighbor_radius must be non-negative")
        if self.multi_event.max_candidates_per_video < 1:
            raise ValueError("multi_event.max_candidates_per_video must be positive")
        if self.multi_event.uniform_fallback_videos < 1:
            raise ValueError("multi_event.uniform_fallback_videos must be positive")
        if self.multi_event.uniform_fallback_frames < 2:
            raise ValueError("multi_event.uniform_fallback_frames must be at least 2")
        if (
            self.multi_event.uniform_fallback_max_candidates
            < self.multi_event.uniform_fallback_frames
        ):
            raise ValueError(
                "multi_event.uniform_fallback_max_candidates must be at least "
                "uniform_fallback_frames"
            )
        if self.qa.result_count != 5:
            raise ValueError("qa.result_count is fixed at 5")
        if self.qa.retrieval_shortlist_size < self.qa.candidate_videos:
            raise ValueError(
                "qa.retrieval_shortlist_size must be at least qa.candidate_videos"
            )
        if self.qa.candidate_videos != self.qa.result_count:
            raise ValueError(
                "qa.candidate_videos must equal result_count so every video is assessed"
            )
        if self.qa.max_rounds < 1:
            raise ValueError("qa.max_rounds must be positive")
        if self.qa.retrieval_shortlist_size < (
            self.qa.candidate_videos * self.qa.max_rounds
        ):
            raise ValueError(
                "qa.retrieval_shortlist_size must cover candidate_videos * max_rounds"
            )
        if self.qa.screening_videos < self.qa.candidate_videos * self.qa.max_rounds:
            raise ValueError(
                "qa.screening_videos must cover candidate_videos * max_rounds"
            )
        if self.qa.retrieval_shortlist_size < self.qa.screening_videos:
            raise ValueError(
                "qa.retrieval_shortlist_size must be at least screening_videos"
            )
        if self.qa.screening_frames_per_video < 1:
            raise ValueError("qa.screening_frames_per_video must be positive")
        if self.qa.max_retrieval_criteria < 1:
            raise ValueError("qa.max_retrieval_criteria must be positive")
        if self.qa.top_frames_per_criterion_per_video < 1:
            raise ValueError(
                "qa.top_frames_per_criterion_per_video must be positive"
            )
        if self.qa.coverage_rank_cutoff < 1:
            raise ValueError("qa.coverage_rank_cutoff must be positive")
        qa_weights = (
            self.qa.coverage_weight,
            self.qa.evidence_score_weight,
            self.qa.main_query_weight,
        )
        if any(weight < 0 for weight in qa_weights) or not abs(sum(qa_weights) - 1.0) < 1e-6:
            raise ValueError(
                "qa coverage/evidence/main weights must be non-negative and sum to 1"
            )
        if self.qa.representatives_per_video < 2:
            raise ValueError("qa.representatives_per_video must be at least 2")
        if self.qa.max_candidates_per_video < self.qa.representatives_per_video:
            raise ValueError(
                "qa.max_candidates_per_video must be at least representatives_per_video"
            )
        if not 0 <= self.qa.minimum_confidence <= 1:
            raise ValueError("qa.minimum_confidence must be in [0, 1]")
        if self.qa_long_range.preserve_main_query_videos < 1:
            raise ValueError(
                "qa_long_range.preserve_main_query_videos must be positive"
            )
        if self.qa_long_range.max_event_queries < 1:
            raise ValueError("qa_long_range.max_event_queries must be positive")
        if self.qa_long_range.screening_frames_per_video < 1:
            raise ValueError(
                "qa_long_range.screening_frames_per_video must be positive"
            )
        if self.qa_long_range.representatives_per_video < 2:
            raise ValueError(
                "qa_long_range.representatives_per_video must be at least 2"
            )
        if (
            self.qa_long_range.max_candidates_per_video
            < self.qa_long_range.representatives_per_video
        ):
            raise ValueError(
                "qa_long_range.max_candidates_per_video must cover representatives"
            )
        if self.temporal_kis.candidate_videos < 1:
            raise ValueError("temporal_kis.candidate_videos must be positive")
        if self.temporal_kis.max_rounds < 1:
            raise ValueError("temporal_kis.max_rounds must be positive")
        if self.temporal_kis.retrieval_shortlist_size < (
            self.temporal_kis.candidate_videos * self.temporal_kis.max_rounds
        ):
            raise ValueError(
                "temporal_kis.retrieval_shortlist_size must cover all rounds"
            )
        if self.temporal_kis.screening_videos < (
            self.temporal_kis.candidate_videos * self.temporal_kis.max_rounds
        ):
            raise ValueError("temporal_kis.screening_videos must cover all rounds")
        if (
            self.temporal_kis.retrieval_shortlist_size
            < self.temporal_kis.screening_videos
        ):
            raise ValueError(
                "temporal_kis.retrieval_shortlist_size must cover screening_videos"
            )
        if self.temporal_kis.screening_frames_per_video < 1:
            raise ValueError(
                "temporal_kis.screening_frames_per_video must be positive"
            )
        if self.temporal_kis.max_retrieval_criteria < 1:
            raise ValueError("temporal_kis.max_retrieval_criteria must be positive")
        if self.temporal_kis.top_frames_per_criterion_per_video < 1:
            raise ValueError(
                "temporal_kis.top_frames_per_criterion_per_video must be positive"
            )
        if self.temporal_kis.coverage_rank_cutoff < 1:
            raise ValueError("temporal_kis.coverage_rank_cutoff must be positive")
        temporal_kis_weights = (
            self.temporal_kis.coverage_weight,
            self.temporal_kis.evidence_score_weight,
            self.temporal_kis.main_query_weight,
        )
        if any(weight < 0 for weight in temporal_kis_weights) or not abs(
            sum(temporal_kis_weights) - 1.0
        ) < 1e-6:
            raise ValueError(
                "temporal_kis coverage/evidence/main weights must sum to 1"
            )
        if self.temporal_kis.representatives_per_video < 2:
            raise ValueError(
                "temporal_kis.representatives_per_video must be at least 2"
            )
        if (
            self.temporal_kis.max_candidates_per_video
            < self.temporal_kis.representatives_per_video
        ):
            raise ValueError(
                "temporal_kis.max_candidates_per_video must cover representatives"
            )
        if not 0 <= self.temporal_kis.minimum_confidence <= 1:
            raise ValueError("temporal_kis.minimum_confidence must be in [0, 1]")
        if self.runtime.task_type_override not in {"auto", "kis", "trake", "qa"}:
            raise ValueError("runtime.task_type_override must be auto, kis, trake or qa")
        if self.runtime.qa_mode_override not in {
            "auto",
            "single_frame",
            "short_window",
            "long_range_temporal",
        }:
            raise ValueError(
                "runtime.qa_mode_override must be auto, single_frame, short_window "
                "or long_range_temporal"
            )
        if not 0 <= self.rescue_zoom.min_primary_score <= 100:
            raise ValueError("rescue_zoom.min_primary_score must be in [0, 100]")
        if self.rescue_zoom.preserve_primary_videos < 1:
            raise ValueError("rescue_zoom.preserve_primary_videos must be positive")
        if self.rescue_zoom.max_candidates < 1:
            raise ValueError("rescue_zoom.max_candidates must be positive")
        if self.rescue_zoom.neighbor_radius < 0:
            raise ValueError("rescue_zoom.neighbor_radius must be non-negative")
        if self.rescue_zoom.cards_per_sheet < 1:
            raise ValueError("rescue_zoom.cards_per_sheet must be positive")
        if not set(self.rescue_zoom.text_detail_regions).issubset(
            {"top", "center", "bottom"}
        ):
            raise ValueError(
                "rescue_zoom.text_detail_regions accepts only top, center, bottom"
            )
        if self.fallback.candidate_videos < 1:
            raise ValueError("fallback.candidate_videos must be positive")
        if self.fallback.relaxed_shortlist_size < self.fallback.candidate_videos:
            raise ValueError(
                "fallback.relaxed_shortlist_size must be at least candidate_videos"
            )
        if self.fallback.representative_frames_per_video < 1:
            raise ValueError("fallback.representative_frames_per_video must be positive")
        if self.fallback.selected_videos_for_deep_scan < 1:
            raise ValueError("fallback.selected_videos_for_deep_scan must be positive")
        if self.fallback.preserve_primary_videos < 0:
            raise ValueError("fallback.preserve_primary_videos must be non-negative")
        if self.fallback.local_scan_anchors_per_video < 1:
            raise ValueError("fallback.local_scan_anchors_per_video must be positive")
        if self.fallback.local_scan_radius < 0:
            raise ValueError("fallback.local_scan_radius must be non-negative")
        if self.fallback.deep_scan_batch_size < 1:
            raise ValueError("fallback.deep_scan_batch_size must be positive")
        if self.fallback.deep_scan_keep_per_video < 1:
            raise ValueError("fallback.deep_scan_keep_per_video must be positive")


def _expand_path(value: str | Path, base: Path | None = None) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute() and base is not None:
        expanded = base / expanded
    return expanded.resolve()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_raw = _section(raw, "data")
    root = _expand_path(data_raw.get("root", config_path.parent), config_path.parent)

    data = DataConfig(
        root=root,
        feature_zip=_expand_path(data_raw["feature_zip"], root),
        mapping_zip=_expand_path(data_raw["mapping_zip"], root),
        keyframe_zip_glob=str(data_raw.get("keyframe_zip_glob", "Keyframes_*.zip")),
        video_zip_glob=str(data_raw.get("video_zip_glob", "Videos_*.zip")),
        cache_dir=_expand_path(data_raw.get("cache_dir", "cache"), config_path.parent),
        output_dir=_expand_path(data_raw.get("output_dir", "outputs"), config_path.parent),
    )
    config = AppConfig(
        data=data,
        retrieval=RetrievalConfig(**_section(raw, "retrieval")),
        contact_sheet=ContactSheetConfig(**_section(raw, "contact_sheet")),
        openai=OpenAIConfig(**_section(raw, "openai")),
        refinement=RefinementConfig(**_section(raw, "refinement")),
        verification=VerificationConfig(**_section(raw, "verification")),
        multi_event=MultiEventConfig(**_section(raw, "multi_event")),
        qa=QAConfig(**_section(raw, "qa")),
        qa_long_range=QALongRangeConfig(**_section(raw, "qa_long_range")),
        temporal_kis=TemporalKISConfig(**_section(raw, "temporal_kis")),
        rescue_zoom=RescueZoomConfig(**_section(raw, "rescue_zoom")),
        fallback=FallbackConfig(**_section(raw, "fallback")),
        runtime=RuntimeConfig(**_section(raw, "runtime")),
    )
    config.validate()
    config.data.cache_dir.mkdir(parents=True, exist_ok=True)
    config.data.output_dir.mkdir(parents=True, exist_ok=True)
    return config
