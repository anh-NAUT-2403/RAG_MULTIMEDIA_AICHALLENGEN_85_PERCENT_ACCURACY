from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .api import OpenAIKeyframeClient
from .config import AppConfig, ContactSheetConfig
from .models import Candidate, EventPlan, QueryPlan, SampleFrame
from .retrieval import CLIPTextEncoder, FeatureZipRetriever
from .storage import KeyframeMapping, SelectiveAssetStore
from .utils import make_run_dir, write_json
from .visual import ContactSheetBuilder, VideoSampler, ZoomEvidenceBuilder


class KeyframeSearchPipeline:
    """Retrieve a short list first, then recover with a video-level scan if needed."""

    def __init__(
        self,
        config: AppConfig,
        api_client: OpenAIKeyframeClient | None = None,
        encoder: CLIPTextEncoder | None = None,
    ) -> None:
        self.config = config
        self.mapping = KeyframeMapping(config.data.mapping_zip)
        self.assets = SelectiveAssetStore(
            data_root=config.data.root,
            keyframe_zip_glob=config.data.keyframe_zip_glob,
            video_zip_glob=config.data.video_zip_glob,
            cache_dir=config.data.cache_dir,
            mapping=self.mapping,
        )
        self.encoder = encoder or CLIPTextEncoder(
            config.retrieval.clip_model, config.retrieval.device
        )
        self.retriever = FeatureZipRetriever(
            feature_zip=config.data.feature_zip,
            top_per_video=config.retrieval.top_per_video,
            shortlist_size=config.retrieval.shortlist_size,
            max_candidates_per_video=config.retrieval.max_candidates_per_video,
        )
        self.sheets = ContactSheetBuilder(config.contact_sheet)
        zoom = config.rescue_zoom
        self.zoom_sheets = ContactSheetBuilder(
            ContactSheetConfig(
                columns=zoom.columns,
                cards_per_sheet=zoom.cards_per_sheet,
                card_width=zoom.card_width,
                card_height=zoom.card_height,
                image_height=zoom.image_height,
                jpeg_quality=config.contact_sheet.jpeg_quality,
            )
        )
        self.zoom_evidence = ZoomEvidenceBuilder(zoom.text_detail_regions)
        self.api = api_client

    def _client(self) -> OpenAIKeyframeClient:
        if self.api is None:
            self.api = OpenAIKeyframeClient(self.config.openai)
        return self.api

    def _prepare_plan(self, query: str, retrieval_only: bool) -> QueryPlan:
        if retrieval_only:
            return QueryPlan(
                original_query=query,
                main_visual_query_en=query,
                must_have=[query],
                search_notes="Retrieval-only mode: query planner was skipped.",
            )
        return self._client().plan_query(
            query, self.config.retrieval.max_support_queries
        )

    def _materialize(self, candidates: list[Candidate]) -> list[Candidate]:
        ready: list[Candidate] = []
        for candidate in candidates:
            try:
                self.mapping.enrich(candidate)
                self.assets.extract_keyframe(candidate)
            except (FileNotFoundError, KeyError, IndexError, ValueError) as error:
                print(f"[skip] {candidate.candidate_id}: {error}")
                continue
            ready.append(candidate)
        return ready

    def _candidate_sheets(
        self, candidates: list[Candidate], directory: Path, prefix: str
    ) -> list[Path]:
        cards = [self.sheets.candidate_card(candidate) for candidate in candidates]
        return self.sheets.build(cards, directory, prefix)

    @staticmethod
    def _selected_candidates(
        ranking: dict[str, Any], candidates: list[Candidate], limit: int
    ) -> list[Candidate]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        selected: list[Candidate] = []
        for row in ranking.get("selected", []):
            candidate = by_id.get(str(row.get("candidate_id", "")))
            if candidate is None or candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return selected

    def _retrieve_and_compare(
        self, query: str, plan: QueryPlan, run_dir: Path
    ) -> tuple[list[Candidate], list[dict[str, Any]], list[Candidate]]:
        support_queries = list(plan.support_queries_en)
        all_rankings: list[dict[str, Any]] = []
        candidate_history: list[Candidate] = []
        best_visual: list[Candidate] = []

        for round_number in range(1, self.config.retrieval.max_rounds + 1):
            supports = support_queries[: self.config.retrieval.max_support_queries]
            retrieval_texts = [plan.main_visual_query_en] + supports
            embeddings = self.encoder.encode(retrieval_texts)
            candidates = self.retriever.retrieve(
                text_embeddings=embeddings,
                main_weight=self.config.retrieval.main_weight,
                support_weight=self.config.retrieval.support_weight,
                source_round=round_number,
            )
            candidates = self._materialize(candidates)
            candidate_history.extend(candidates)
            round_dir = run_dir / f"round_{round_number:02d}"
            sheets = self._candidate_sheets(candidates, round_dir, "retrieval")
            write_json(round_dir / "candidates.json", [item.to_dict() for item in candidates])

            ranking = self._client().rank_contact_sheets(
                query,
                plan,
                candidates,
                sheets,
                self.config.refinement.finalists,
            )
            all_rankings.append(ranking)
            write_json(round_dir / "vlm_ranking.json", ranking)
            selected = self._selected_candidates(
                ranking, candidates, self.config.refinement.finalists
            )
            if selected:
                best_visual = selected

            top_score = max(
                (int(row.get("match_score", 0)) for row in ranking.get("selected", [])),
                default=0,
            )
            if ranking.get("sufficient_match") and (
                top_score >= self.config.retrieval.visual_accept_score
            ):
                break

            refined = str(ranking.get("refined_retrieval_query_en", "")).strip()
            if not refined or round_number >= self.config.retrieval.max_rounds:
                break
            # Feedback is weak support. The faithful main query remains authoritative.
            support_queries = [refined] + plan.support_queries_en

        return best_visual, all_rankings, candidate_history

    def _motion_refine_one(
        self,
        query: str,
        plan: QueryPlan,
        center: Candidate,
        run_dir: Path,
    ) -> Candidate:
        cfg = self.config.refinement
        video_path = self.assets.extract_video(center.video_id)
        sample_dir = run_dir / "motion" / center.candidate_id / "frames"
        sampler = VideoSampler()
        samples = sampler.sample(
            video_path=video_path,
            video_id=center.video_id,
            center_time=float(center.pts_time or 0.0),
            before=cfg.video_window_before,
            after=cfg.video_window_after,
            fps=cfg.motion_fps if plan.needs_motion else cfg.static_fps,
            max_frames=cfg.max_sampled_frames,
            output_dir=sample_dir,
        )
        cards = [self.sheets.sample_card(sample) for sample in samples]
        sheet_paths = self.sheets.build(
            cards, sample_dir.parent, f"{center.video_id}_samples"
        )
        write_json(sample_dir.parent / "samples.json", [item.to_dict() for item in samples])
        decision = self._client().select_sample(query, plan, samples, sheet_paths)
        write_json(sample_dir.parent / "vlm_sample_selection.json", decision)
        selected_id = decision.get("selected_sample_id")
        selected = next((item for item in samples if item.sample_id == selected_id), None)
        if selected is None:
            return center
        nearest = self.mapping.nearest_time(center.video_id, selected.pts_time)
        nearest.retrieval_score = center.retrieval_score
        nearest.main_score = center.main_score
        nearest.support_scores = center.support_scores
        nearest.source_round = center.source_round
        self.assets.extract_keyframe(nearest)
        return nearest

    def _motion_refine(
        self,
        query: str,
        plan: QueryPlan,
        finalists: list[Candidate],
        run_dir: Path,
    ) -> list[Candidate]:
        cfg = self.config.refinement
        if not (cfg.use_raw_video_for_motion and plan.needs_motion):
            return finalists
        refined: list[Candidate] = []
        processed_videos: set[str] = set()
        for index, candidate in enumerate(finalists):
            if (
                index >= cfg.raw_video_finalists
                or candidate.video_id in processed_videos
            ):
                refined.append(candidate)
                continue
            try:
                processed_videos.add(candidate.video_id)
                refined.append(
                    self._motion_refine_one(query, plan, candidate, run_dir)
                )
            except (FileNotFoundError, RuntimeError) as error:
                print(f"[motion fallback] {candidate.candidate_id}: {error}")
                refined.append(candidate)
        if not self.config.runtime.keep_extracted_videos:
            for video_id in processed_videos:
                video_path = self.config.data.cache_dir / "videos" / f"{video_id}.mp4"
                if video_path.is_file():
                    video_path.unlink()
        return refined

    def _neighbor_candidates(self, centers: list[Candidate]) -> list[Candidate]:
        by_id: dict[str, Candidate] = {}
        for center in centers:
            for neighbor in self.assets.neighbors(
                center, self.config.refinement.neighbor_radius
            ):
                current = by_id.get(neighbor.candidate_id)
                if current is None or neighbor.retrieval_score > current.retrieval_score:
                    by_id[neighbor.candidate_id] = neighbor
        return sorted(
            by_id.values(),
            key=lambda item: (item.retrieval_score, -abs(item.ordinal)),
            reverse=True,
        )

    @staticmethod
    def _event_query_plan(query: str, event: EventPlan) -> QueryPlan:
        must_have = [
            criterion.description
            for criterion in event.criteria
            if criterion.required and criterion.criterion_type != "context"
        ]
        return QueryPlan(
            original_query=event.description_vi or query,
            main_visual_query_en=event.visual_query_en,
            must_have=must_have,
            needs_motion=any(
                criterion.required and criterion.criterion_type == "temporal"
                for criterion in event.criteria
            ),
            needs_text_reading=any(
                criterion.required and criterion.criterion_type == "exact_text"
                for criterion in event.criteria
            ),
            criteria=event.criteria,
        )

    def _prepare_multi_event_candidates(
        self, plan: QueryPlan, run_dir: Path
    ) -> tuple[dict[str, list[Candidate]], list[dict[str, object]]]:
        cfg = self.config.multi_event
        embeddings = self.encoder.encode(
            [event.visual_query_en for event in plan.events]
        )
        event_hits, video_ranking = self.retriever.retrieve_multi_event(
            embeddings,
            candidate_videos=cfg.candidate_videos,
            top_frames_per_event_per_video=cfg.top_frames_per_event_per_video,
        )
        write_json(run_dir / "video_ranking.json", video_ranking)
        prepared: dict[str, list[Candidate]] = {}
        for video_id, per_event in event_hits.items():
            by_id: dict[str, Candidate] = {}
            for centers in per_event:
                ready_centers = self._materialize(centers)
                for center in ready_centers:
                    for candidate in self.assets.neighbors(center, cfg.neighbor_radius):
                        current = by_id.get(candidate.candidate_id)
                        if (
                            current is None
                            or candidate.retrieval_score > current.retrieval_score
                        ):
                            by_id[candidate.candidate_id] = candidate
            # Keep temporal order on contact sheets so sequence judgments are easier.
            candidates = sorted(by_id.values(), key=lambda item: item.ordinal)
            if len(candidates) > cfg.max_candidates_per_video:
                strongest = sorted(
                    candidates, key=lambda item: item.retrieval_score, reverse=True
                )[: cfg.max_candidates_per_video]
                candidates = sorted(strongest, key=lambda item: item.ordinal)
            prepared[video_id] = candidates
        return prepared, video_ranking

    def _normalize_multi_event_decision(
        self,
        query: str,
        plan: QueryPlan,
        video_id: str,
        candidates: list[Candidate],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        rows_by_id = {
            str(row.get("event_id", "")): row
            for row in decision.get("events", [])
            if isinstance(row, dict)
        }
        normalized: list[dict[str, Any]] = []
        selected_times: list[float] = []
        all_events_verified = True
        for event in plan.events:
            row = rows_by_id.get(event.event_id)
            if row is None:
                row = {
                    "event_id": event.event_id,
                    "selected_candidate_id": None,
                    "verified": False,
                    "confidence": 0.0,
                    "satisfied": [],
                    "missing": [event.description_vi],
                    "reason": "The VLM did not return this requested event.",
                    "criterion_results": [],
                    "ranked_candidates": [],
                }
            event_plan = self._event_query_plan(query, event)
            self._apply_hard_gate(event_plan, row)
            self._enrich_ranked_candidates(row, candidates)
            selected_id = str(row.get("selected_candidate_id") or "")
            selected = next(
                (
                    item
                    for item in row.get("ranked_candidates", [])
                    if item.get("candidate_id") == selected_id
                ),
                None,
            )
            event_verified = bool(row.get("verified")) and selected is not None
            criteria_by_id = {
                criterion.criterion_id: criterion for criterion in event.criteria
            }
            visible_match = any(
                bool(result.get("satisfied"))
                and float(result.get("confidence", 0.0))
                >= self.config.verification.criterion_confidence_threshold
                and criteria_by_id.get(str(result.get("criterion_id", "")))
                is not None
                and criteria_by_id[str(result.get("criterion_id", ""))].criterion_type
                in {"visual", "exact_text", "text_presence", "count"}
                for result in row.get("criterion_results", [])
                if isinstance(result, dict)
            )
            event_matched = selected is not None or visible_match
            if selected is not None:
                selected_times.append(float(selected.get("pts_time") or 0.0))
            else:
                selected_times.append(float("inf"))
            all_events_verified = all_events_verified and event_verified
            normalized.append(
                {
                    "event_id": event.event_id,
                    "description": event.description_vi,
                    "verified": event_verified,
                    "matched": event_matched,
                    "confidence": float(row.get("confidence", 0.0)),
                    "reason": row.get("reason", ""),
                    "satisfied": row.get("satisfied", []),
                    "missing": row.get("missing", []),
                    "criterion_results": row.get("criterion_results", []),
                    "hard_gate_passed": row.get("hard_gate_passed"),
                    "hard_gate_failures": row.get("hard_gate_failures", []),
                    "selected": selected,
                    "alternatives": row.get("ranked_candidates", []),
                }
            )

        chronological = all(
            earlier < later
            for earlier, later in zip(selected_times, selected_times[1:])
        )
        model_verified = bool(decision.get("verified", False))
        verified = model_verified and all_events_verified and chronological
        confidence = min(
            (float(row["confidence"]) for row in normalized), default=0.0
        )
        event_coverage = sum(bool(row["matched"]) for row in normalized)
        return {
            "verified": verified,
            "model_verified": model_verified,
            "task_type": "trake",
            "strategy": "multi_event_trake",
            "query": query,
            "video_id": video_id,
            "confidence": confidence,
            "event_coverage": event_coverage,
            "event_count": len(normalized),
            "chronological": chronological,
            "reason": decision.get("reason", ""),
            "events": normalized,
        }

    def _run_multi_event_uniform_fallback(
        self,
        query: str,
        plan: QueryPlan,
        run_dir: Path,
        video_id: str,
        original_candidates: list[Candidate],
    ) -> dict[str, Any]:
        """Inspect one promising video's full timeline when CLIP misses an event."""
        cfg = self.config.multi_event
        total = len(self.mapping.rows(video_id))
        uniform = self._materialize_video_ordinals(
            video_id,
            self._uniform_ordinals(total, cfg.uniform_fallback_frames),
            score=0.0,
            source_round=300,
        )
        uniform_ids = {candidate.candidate_id for candidate in uniform}
        candidates = list(uniform)
        remaining = max(0, cfg.uniform_fallback_max_candidates - len(candidates))
        strongest_original = sorted(
            (
                candidate
                for candidate in original_candidates
                if candidate.candidate_id not in uniform_ids
            ),
            key=lambda item: item.retrieval_score,
            reverse=True,
        )[:remaining]
        candidates.extend(strongest_original)
        candidates.sort(key=lambda item: item.ordinal)

        fallback_dir = run_dir / "multi_event_fallback" / video_id
        write_json(
            fallback_dir / "candidates.json",
            [candidate.to_dict() for candidate in candidates],
        )
        sheets = self._candidate_sheets(candidates, fallback_dir, "uniform_sequence")
        raw_decision = self._client().select_multi_event_sequence(
            query, plan, video_id, candidates, sheets
        )
        write_json(fallback_dir / "vlm_sequence_raw.json", raw_decision)
        normalized = self._normalize_multi_event_decision(
            query, plan, video_id, candidates, raw_decision
        )
        normalized["scan_stage"] = "uniform_fallback"
        write_json(fallback_dir / "vlm_sequence_normalized.json", normalized)
        return normalized

    def _run_multi_event(self, query: str, plan: QueryPlan, run_dir: Path) -> dict[str, Any]:
        candidates_by_video, video_ranking = self._prepare_multi_event_candidates(
            plan, run_dir
        )
        attempts: list[dict[str, Any]] = []
        for video_id, candidates in candidates_by_video.items():
            video_dir = run_dir / "multi_event" / video_id
            write_json(video_dir / "candidates.json", [item.to_dict() for item in candidates])
            sheets = self._candidate_sheets(candidates, video_dir, "sequence")
            raw_decision = self._client().select_multi_event_sequence(
                query, plan, video_id, candidates, sheets
            )
            write_json(video_dir / "vlm_sequence_raw.json", raw_decision)
            normalized = self._normalize_multi_event_decision(
                query, plan, video_id, candidates, raw_decision
            )
            normalized["scan_stage"] = "clip_candidates"
            write_json(video_dir / "vlm_sequence_normalized.json", normalized)
            attempts.append(normalized)

        if (
            attempts
            and not any(bool(item["verified"]) for item in attempts)
            and self.config.multi_event.uniform_fallback_enabled
        ):
            promising = sorted(
                attempts,
                key=lambda item: (
                    int(item.get("event_coverage", 0)),
                    float(item.get("confidence", 0.0)),
                ),
                reverse=True,
            )[: self.config.multi_event.uniform_fallback_videos]
            for attempt in promising:
                video_id = str(attempt["video_id"])
                fallback_attempt = self._run_multi_event_uniform_fallback(
                    query,
                    plan,
                    run_dir,
                    video_id,
                    candidates_by_video.get(video_id, []),
                )
                attempts.append(fallback_attempt)

        if attempts:
            best_attempt = max(
                attempts,
                key=lambda item: (
                    bool(item["verified"]),
                    int(item.get("event_coverage", 0)),
                    bool(item.get("chronological", False)),
                    float(item["confidence"]),
                ),
            )
            # Detach the returned result completely from the attempt object graph.
            # Per-video normalized JSON was already written above, so this is also a
            # useful serialization boundary before adding result-only diagnostics.
            best = json.loads(json.dumps(best_attempt, ensure_ascii=False))
        else:
            best = {
                "verified": False,
                "task_type": "trake",
                "strategy": "multi_event_trake",
                "query": query,
                "video_id": None,
                "confidence": 0.0,
                "chronological": False,
                "reason": "No candidate video was available for multi-event verification.",
                "events": [],
            }
        keyframes: list[dict[str, Any]] = []
        for event in best.get("events", []):
            selected = event.get("selected")
            if selected:
                keyframes.append({"event_id": event["event_id"], **selected})
        best["keyframes"] = keyframes
        best["video_ranking"] = json.loads(
            json.dumps(video_ranking, ensure_ascii=False)
        )
        # Keep diagnostics useful but compact and JSON-safe. Full per-video details
        # remain in multi_event/<video_id>/vlm_sequence_normalized.json.
        best["video_attempts"] = [
            {
                "video_id": item.get("video_id"),
                "verified": item.get("verified"),
                "confidence": item.get("confidence"),
                "event_coverage": item.get("event_coverage"),
                "event_count": item.get("event_count"),
                "chronological": item.get("chronological"),
                "scan_stage": item.get("scan_stage"),
                "reason": item.get("reason"),
            }
            for item in attempts
        ]
        best["run_dir"] = str(run_dir)
        write_json(run_dir / "result.json", best)
        self._write_result_csv(run_dir / "result.csv", best)
        return best

    def _attempt_final_selection(
        self,
        query: str,
        plan: QueryPlan,
        centers: list[Candidate],
        artifact_dir: Path,
    ) -> tuple[Candidate | None, dict[str, Any]]:
        if not centers:
            return None, {
                "verified": False,
                "selected_candidate_id": None,
                "confidence": 0.0,
                "satisfied": [],
                "missing": [],
                "reason": "No candidate centers were available for final verification.",
                "ranked_candidates": [],
            }
        centers = self._motion_refine(query, plan, centers, artifact_dir)
        neighbors = self._neighbor_candidates(centers)
        if not neighbors:
            return None, {
                "verified": False,
                "selected_candidate_id": None,
                "confidence": 0.0,
                "satisfied": [],
                "missing": [],
                "reason": "No neighboring keyframes could be extracted.",
                "ranked_candidates": [],
            }
        neighbor_dir = artifact_dir / "final_neighbors"
        neighbor_sheets = self._candidate_sheets(neighbors, neighbor_dir, "neighbors")
        write_json(neighbor_dir / "candidates.json", [item.to_dict() for item in neighbors])
        decision = self._client().select_final(query, plan, neighbors, neighbor_sheets)
        self._apply_hard_gate(plan, decision)
        self._enrich_ranked_candidates(decision, neighbors)
        write_json(neighbor_dir / "vlm_final_selection.json", decision)
        selected_id = decision.get("selected_candidate_id")
        selected = next(
            (item for item in neighbors if item.candidate_id == selected_id), None
        )
        return selected, decision

    @staticmethod
    def _enrich_ranked_candidates(
        decision: dict[str, Any], candidates: list[Candidate]
    ) -> None:
        """Attach frame metadata to VLM rankings and support older API mocks."""
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()

        rows = decision.get("ranked_candidates", [])
        if not isinstance(rows, list):
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("candidate_id", ""))
            candidate = by_id.get(candidate_id)
            if candidate is None or candidate_id in seen:
                continue
            seen.add(candidate_id)
            ranked.append(
                {
                    "candidate_id": candidate_id,
                    "video_id": candidate.video_id,
                    "frame_idx": candidate.frame_idx,
                    "keyframe_ordinal": candidate.ordinal,
                    "pts_time": candidate.pts_time,
                    "fps": candidate.fps,
                    "verified": bool(row.get("verified", False)),
                    "confidence": float(row.get("confidence", 0.0)),
                    "satisfied": row.get("satisfied", []),
                    "missing": row.get("missing", []),
                    "reason": row.get("reason", ""),
                    "criterion_results": row.get("criterion_results", []),
                    "hard_gate_passed": row.get("hard_gate_passed"),
                    "hard_gate_failures": row.get("hard_gate_failures", []),
                    "keyframe_path": str(candidate.image_path),
                }
            )

        # Old test clients and cached decisions only contain selected_candidate_id.
        # Preserve backward compatibility by synthesizing the first ranked row.
        selected_id = str(decision.get("selected_candidate_id") or "")
        selected = by_id.get(selected_id)
        if selected is not None:
            selected_row = next(
                (row for row in ranked if row["candidate_id"] == selected_id), None
            )
            if selected_row is not None:
                ranked.remove(selected_row)
                # The top-level decision is authoritative for the selected row.
                selected_row.update(
                    {
                        "verified": bool(decision.get("verified", False)),
                        "confidence": float(decision.get("confidence", 0.0)),
                        "satisfied": decision.get("satisfied", []),
                        "missing": decision.get("missing", []),
                        "reason": decision.get("reason", ""),
                        "criterion_results": decision.get("criterion_results", []),
                        "hard_gate_passed": decision.get("hard_gate_passed"),
                        "hard_gate_failures": decision.get(
                            "hard_gate_failures", []
                        ),
                    }
                )
            else:
                selected_row = {
                    "candidate_id": selected_id,
                    "video_id": selected.video_id,
                    "frame_idx": selected.frame_idx,
                    "keyframe_ordinal": selected.ordinal,
                    "pts_time": selected.pts_time,
                    "fps": selected.fps,
                    "verified": bool(decision.get("verified", False)),
                    "confidence": float(decision.get("confidence", 0.0)),
                    "satisfied": decision.get("satisfied", []),
                    "missing": decision.get("missing", []),
                    "reason": decision.get("reason", ""),
                    "criterion_results": decision.get("criterion_results", []),
                    "hard_gate_passed": decision.get("hard_gate_passed"),
                    "hard_gate_failures": decision.get("hard_gate_failures", []),
                    "keyframe_path": str(selected.image_path),
                }
            ranked.insert(0, selected_row)
        decision["ranked_candidates"] = ranked

    def _apply_hard_gate(self, plan: QueryPlan, decision: dict[str, Any]) -> None:
        """Recompute verification from generic typed criteria, not query words."""
        cfg = self.config.verification
        hard_criteria = [
            criterion
            for criterion in plan.criteria
            if criterion.required
            and criterion.criterion_type in cfg.hard_gate_types
            and criterion.evidence_scope == "frame"
            and criterion.classification_confidence
            >= cfg.classification_confidence_threshold
        ]

        def apply(target: dict[str, Any]) -> None:
            model_verified = bool(target.get("verified", False))
            target["model_verified"] = model_verified
            if not cfg.enabled or not hard_criteria:
                target["hard_gate_passed"] = True
                target["hard_gate_failures"] = []
                return

            raw_results = target.get("criterion_results", [])
            results = {
                str(item.get("criterion_id", "")): item
                for item in raw_results
                if isinstance(item, dict)
            }
            failures: list[dict[str, Any]] = []
            for criterion in hard_criteria:
                evidence = results.get(criterion.criterion_id)
                passed = bool(evidence and evidence.get("satisfied")) and (
                    float(evidence.get("confidence", 0.0))
                    >= cfg.criterion_confidence_threshold
                )
                if not passed:
                    failures.append(
                        {
                            "criterion_id": criterion.criterion_id,
                            "criterion_type": criterion.criterion_type,
                            "description": criterion.description,
                            "evidence": evidence,
                        }
                    )
            target["hard_gate_passed"] = not failures
            target["hard_gate_failures"] = failures
            target["verified"] = model_verified and not failures

        apply(decision)
        rows = decision.get("ranked_candidates", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    apply(row)

    @staticmethod
    def _is_verified(
        selected: Candidate | None, decision: dict[str, Any], threshold: float
    ) -> bool:
        return bool(selected) and bool(decision.get("verified")) and (
            float(decision.get("confidence", 0.0)) >= threshold
        )

    def _strong_primary_anchors(
        self,
        rankings: list[dict[str, Any]],
        candidate_history: list[Candidate],
    ) -> list[tuple[int, Candidate]]:
        """Keep VLM-supported primary evidence instead of letting fallback forget it."""
        cfg = self.config.rescue_zoom
        by_id = {candidate.candidate_id: candidate for candidate in candidate_history}
        best_by_candidate: dict[str, int] = {}
        for ranking in rankings:
            for row in ranking.get("selected", []):
                candidate_id = str(row.get("candidate_id", ""))
                score = int(row.get("match_score", 0))
                if candidate_id in by_id:
                    best_by_candidate[candidate_id] = max(
                        score, best_by_candidate.get(candidate_id, 0)
                    )

        by_video: dict[str, list[tuple[int, Candidate]]] = {}
        for candidate_id, score in best_by_candidate.items():
            candidate = by_id[candidate_id]
            by_video.setdefault(candidate.video_id, []).append((score, candidate))
        ordered_videos = sorted(
            (
                item
                for item in by_video.items()
                if max(score for score, _ in item[1]) >= cfg.min_primary_score
            ),
            key=lambda item: max(score for score, _ in item[1]),
            reverse=True,
        )[: cfg.preserve_primary_videos]
        anchors: list[tuple[int, Candidate]] = []
        for _, items in ordered_videos:
            anchors.extend(sorted(items, key=lambda item: item[0], reverse=True))
        return anchors

    def _rescue_candidates(
        self, scored_anchors: list[tuple[int, Candidate]]
    ) -> list[Candidate]:
        """Round-robin neighboring frames so evidence clusters all survive the cap."""
        cfg = self.config.rescue_zoom
        groups: list[list[Candidate]] = []
        for score, anchor in scored_anchors:
            anchor.retrieval_score = score / 100.0
            anchor.main_score = anchor.retrieval_score
            neighbors = self.assets.neighbors(anchor, cfg.neighbor_radius)
            neighbors.sort(key=lambda item: abs(item.ordinal - anchor.ordinal))
            groups.append(neighbors)

        candidates: list[Candidate] = []
        seen: set[str] = set()
        offset = 0
        max_group_length = max((len(group) for group in groups), default=0)
        while len(candidates) < cfg.max_candidates and offset < max_group_length:
            for group in groups:
                if offset >= len(group):
                    continue
                candidate = group[offset]
                if candidate.candidate_id not in seen:
                    seen.add(candidate.candidate_id)
                    candidates.append(candidate)
                    if len(candidates) >= cfg.max_candidates:
                        break
            offset += 1
        return candidates

    def _attempt_rescue_zoom(
        self,
        query: str,
        plan: QueryPlan,
        rankings: list[dict[str, Any]],
        candidate_history: list[Candidate],
        run_dir: Path,
    ) -> tuple[Candidate | None, dict[str, Any], list[Candidate], dict[str, Any]]:
        cfg = self.config.rescue_zoom
        if not cfg.enabled:
            return None, {
                "verified": False,
                "selected_candidate_id": None,
                "confidence": 0.0,
                "satisfied": [],
                "missing": [],
                "reason": "Rescue Zoom is disabled.",
                "ranked_candidates": [],
            }, [], {"enabled": False}

        scored_anchors = self._strong_primary_anchors(rankings, candidate_history)
        if not scored_anchors:
            return None, {
                "verified": False,
                "selected_candidate_id": None,
                "confidence": 0.0,
                "satisfied": [],
                "missing": [],
                "reason": "No primary candidate met the Rescue Zoom score threshold.",
                "ranked_candidates": [],
            }, [], {"enabled": True, "anchors": []}

        candidates = self._rescue_candidates(scored_anchors)
        rescue_dir = run_dir / "rescue_zoom"
        write_json(rescue_dir / "candidates.json", [item.to_dict() for item in candidates])
        sheets = self.zoom_evidence.build(
            candidates,
            self.zoom_sheets,
            rescue_dir,
            include_detail_crops=(plan.needs_text_reading and cfg.crop_text_when_needed),
        )
        decision = self._client().select_rescue_zoom(query, plan, candidates, sheets)
        self._apply_hard_gate(plan, decision)
        self._enrich_ranked_candidates(decision, candidates)
        write_json(rescue_dir / "vlm_final_selection.json", decision)
        selected_id = decision.get("selected_candidate_id")
        selected = next(
            (item for item in candidates if item.candidate_id == selected_id), None
        )
        summary = {
            "enabled": True,
            "anchors": [
                {"match_score": score, "candidate": candidate.to_dict()}
                for score, candidate in scored_anchors
            ],
            "candidate_count": len(candidates),
            "contact_sheets": [str(path) for path in sheets],
        }
        write_json(rescue_dir / "summary.json", summary)
        return selected, decision, [candidate for _, candidate in scored_anchors], summary

    @staticmethod
    def _unique_video_ids(candidates: list[Candidate], limit: int) -> list[str]:
        video_ids: list[str] = []
        for candidate in candidates:
            if candidate.video_id not in video_ids:
                video_ids.append(candidate.video_id)
            if len(video_ids) >= limit:
                break
        return video_ids

    @staticmethod
    def _uniform_ordinals(total: int, requested: int) -> list[int]:
        if total < 1:
            return []
        count = min(total, requested)
        if count == 1:
            return [max(1, (total + 1) // 2)]
        return sorted(
            {
                round(index * (total - 1) / (count - 1)) + 1
                for index in range(count)
            }
        )

    def _materialize_video_ordinals(
        self,
        video_id: str,
        ordinals: list[int],
        score: float,
        source_round: int,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for ordinal in ordinals:
            try:
                candidate = self.mapping.candidate_at(video_id, ordinal)
            except (KeyError, IndexError, ValueError):
                continue
            candidate.retrieval_score = score
            candidate.main_score = score
            candidate.source_round = source_round
            candidates.append(candidate)
        return self._materialize(candidates)

    def _fallback_seed_videos(
        self,
        plan: QueryPlan,
        candidate_history: list[Candidate],
        fallback_dir: Path,
        preserved_anchors: list[Candidate],
    ) -> list[tuple[str, float]]:
        all_candidates = list(candidate_history)
        if (
            self.config.fallback.relaxed_query_enabled
            and plan.fallback_visual_query_en.strip()
        ):
            embeddings = self.encoder.encode([plan.fallback_visual_query_en])
            # Use one candidate per video and a wider shortlist here. The normal
            # shortlist is optimized for precision, while this branch needs recall.
            fallback_retriever = FeatureZipRetriever(
                feature_zip=self.config.data.feature_zip,
                top_per_video=1,
                shortlist_size=self.config.fallback.relaxed_shortlist_size,
                max_candidates_per_video=1,
            )
            relaxed = fallback_retriever.retrieve(
                text_embeddings=embeddings,
                main_weight=1.0,
                support_weight=0.0,
                source_round=99,
            )
            all_candidates.extend(relaxed)
            write_json(
                fallback_dir / "relaxed_retrieval_candidates.json",
                [item.to_dict() for item in relaxed],
            )

        best_by_video: dict[str, Candidate] = {}
        for candidate in all_candidates:
            current = best_by_video.get(candidate.video_id)
            if current is None or candidate.retrieval_score > current.retrieval_score:
                best_by_video[candidate.video_id] = candidate
        ranked_seeds = sorted(
            (
                (video_id, candidate.retrieval_score)
                for video_id, candidate in best_by_video.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        seeds: list[tuple[str, float]] = []
        for video_id in self._unique_video_ids(
            preserved_anchors, self.config.fallback.preserve_primary_videos
        ):
            if video_id not in best_by_video:
                continue
            seeds.append((video_id, best_by_video[video_id].retrieval_score))
        for item in ranked_seeds:
            if item[0] not in {video_id for video_id, _ in seeds}:
                seeds.append(item)
            if len(seeds) >= self.config.fallback.candidate_videos:
                break
        write_json(
            fallback_dir / "seed_videos.json",
            [
                {
                    "video_id": video_id,
                    "seed_score": score,
                    "best_candidate": best_by_video[video_id].to_dict(),
                }
                for video_id, score in seeds
            ],
        )
        return seeds

    def _video_level_fallback(
        self,
        query: str,
        plan: QueryPlan,
        candidate_history: list[Candidate],
        run_dir: Path,
        preserved_anchors: list[Candidate],
    ) -> tuple[list[Candidate], dict[str, Any]]:
        """Sample candidate videos, then inspect only local windows around anchors."""
        fallback_dir = run_dir / "fallback_video_scan"
        seeds = self._fallback_seed_videos(
            plan, candidate_history, fallback_dir, preserved_anchors
        )
        if not seeds:
            return [], {"reason": "No candidate videos were available for fallback."}

        seed_scores = dict(seeds)
        representatives: list[Candidate] = []
        for video_id, score in seeds:
            try:
                total = len(self.mapping.rows(video_id))
            except (KeyError, ValueError):
                continue
            representatives.extend(
                self._materialize_video_ordinals(
                    video_id,
                    self._uniform_ordinals(
                        total, self.config.fallback.representative_frames_per_video
                    ),
                    score,
                    source_round=100,
                )
            )
        representative_dir = fallback_dir / "representatives"
        if not representatives:
            return [], {"reason": "Representative keyframes could not be extracted."}
        representative_sheets = self._candidate_sheets(
            representatives, representative_dir, "video_representatives"
        )
        write_json(
            representative_dir / "candidates.json",
            [item.to_dict() for item in representatives],
        )
        video_decision = self._client().select_videos(
            query,
            plan,
            representatives,
            representative_sheets,
            self.config.fallback.selected_videos_for_deep_scan,
        )
        write_json(representative_dir / "vlm_video_selection.json", video_decision)

        available = {item.video_id for item in representatives}
        selected_videos: list[str] = []
        for row in video_decision.get("selected", []):
            video_id = str(row.get("video_id", ""))
            if video_id in available and video_id not in selected_videos:
                selected_videos.append(video_id)
        for video_id in reversed(
            self._unique_video_ids(
                preserved_anchors, self.config.fallback.preserve_primary_videos
            )
        ):
            if video_id in available and video_id not in selected_videos:
                selected_videos.insert(0, video_id)
        selected_videos = selected_videos[
            : self.config.fallback.selected_videos_for_deep_scan
        ]
        if not selected_videos:
            return [], {
                "reason": video_decision.get(
                    "reason", "The VLM did not select a video for local scanning."
                ),
                "video_decision": video_decision,
            }

        selected_centers: list[Candidate] = []
        deep_summary: dict[str, Any] = {}
        for video_id in selected_videos:
            try:
                total = len(self.mapping.rows(video_id))
            except (KeyError, ValueError):
                continue
            video_dir = fallback_dir / "local_scan" / video_id
            video_representatives = [
                candidate for candidate in representatives if candidate.video_id == video_id
            ]
            anchor_dir = video_dir / "anchors"
            anchor_sheets = self._candidate_sheets(
                video_representatives, anchor_dir, "representatives"
            )
            anchor_decision = self._client().select_local_anchors(
                query,
                plan,
                video_representatives,
                anchor_sheets,
                self.config.fallback.local_scan_anchors_per_video,
            )
            write_json(anchor_dir / "vlm_anchor_selection.json", anchor_decision)
            anchors: list[Candidate] = []
            for anchor in preserved_anchors:
                if anchor.video_id == video_id and anchor not in anchors:
                    anchors.append(anchor)
                if len(anchors) >= self.config.fallback.local_scan_anchors_per_video:
                    break
            for anchor in self._selected_candidates(
                anchor_decision,
                video_representatives,
                self.config.fallback.local_scan_anchors_per_video,
            ):
                if anchor not in anchors:
                    anchors.append(anchor)
                if len(anchors) >= self.config.fallback.local_scan_anchors_per_video:
                    break
            if not anchors:
                deep_summary[video_id] = {
                    "keyframe_count": total,
                    "local_scan_keyframes": 0,
                    "anchors": [],
                    "selected_centers": [],
                }
                continue

            local_ordinals = sorted(
                {
                    ordinal
                    for anchor in anchors
                    for ordinal in range(
                        max(1, anchor.ordinal - self.config.fallback.local_scan_radius),
                        min(total, anchor.ordinal + self.config.fallback.local_scan_radius) + 1,
                    )
                }
            )
            ranked: list[tuple[int, Candidate]] = []
            for offset in range(0, len(local_ordinals), self.config.fallback.deep_scan_batch_size):
                ordinals = local_ordinals[
                    offset : offset + self.config.fallback.deep_scan_batch_size
                ]
                batch = self._materialize_video_ordinals(
                    video_id, ordinals, seed_scores[video_id], source_round=101
                )
                if not batch:
                    continue
                batch_number = offset // self.config.fallback.deep_scan_batch_size + 1
                batch_dir = video_dir / f"batch_{batch_number:03d}"
                sheets = self._candidate_sheets(batch, batch_dir, "local_scan")
                write_json(batch_dir / "candidates.json", [item.to_dict() for item in batch])
                ranking = self._client().rank_contact_sheets(query, plan, batch, sheets, 1)
                write_json(batch_dir / "vlm_ranking.json", ranking)
                scores = {
                    str(row.get("candidate_id", "")): int(row.get("match_score", 0))
                    for row in ranking.get("selected", [])
                }
                for candidate in self._selected_candidates(ranking, batch, 1):
                    ranked.append((scores.get(candidate.candidate_id, 0), candidate))

            if not ranked:
                ranked = [(0, anchor) for anchor in anchors]

            ranked.sort(key=lambda item: (item[0], item[1].ordinal), reverse=True)
            kept: list[Candidate] = []
            seen: set[str] = set()
            for match_score, candidate in ranked:
                if candidate.candidate_id in seen:
                    continue
                seen.add(candidate.candidate_id)
                # Preserve VLM's deep-scan ordering for the final neighbor comparison.
                candidate.retrieval_score = match_score / 100.0
                candidate.main_score = candidate.retrieval_score
                kept.append(candidate)
                if len(kept) >= self.config.fallback.deep_scan_keep_per_video:
                    break
            selected_centers.extend(kept)
            deep_summary[video_id] = {
                "keyframe_count": total,
                "local_scan_keyframes": len(local_ordinals),
                "anchors": [item.to_dict() for item in anchors],
                "selected_centers": [item.to_dict() for item in kept],
            }

        summary = {
            "selected_videos": selected_videos,
            "video_decision": video_decision,
            "local_scan": deep_summary,
            "reason": "Video-level fallback completed with local scans.",
        }
        write_json(fallback_dir / "summary.json", summary)
        return selected_centers, summary

    @staticmethod
    def _qa_round_robin_candidates(
        candidates: list[Candidate], limit: int
    ) -> list[Candidate]:
        """Keep diverse evidence sources before taking second hits per criterion."""
        groups: dict[int, list[Candidate]] = {}
        for candidate in candidates:
            groups.setdefault(candidate.source_round, []).append(candidate)
        for group in groups.values():
            group.sort(key=lambda item: item.retrieval_score, reverse=True)
        selected: list[Candidate] = []
        seen: set[str] = set()
        offset = 0
        while len(selected) < limit:
            added = False
            for source_round in sorted(groups):
                group = groups[source_round]
                if offset >= len(group):
                    continue
                candidate = group[offset]
                added = True
                if candidate.candidate_id not in seen:
                    seen.add(candidate.candidate_id)
                    selected.append(candidate)
                    if len(selected) >= limit:
                        break
            if not added:
                break
            offset += 1
        return sorted(selected, key=lambda candidate: candidate.ordinal)

    def _prepare_qa_candidates(
        self,
        plan: QueryPlan,
        run_dir: Path,
        coverage_config: Any | None = None,
        artifact_prefix: str = "qa",
        qa_mode: str = "single_frame",
    ) -> tuple[list[Candidate], list[str], list[dict[str, Any]]]:
        """Retrieve independent evidence, then represent a ranked pool of videos."""
        cfg = coverage_config or self.config.qa
        long_range = qa_mode == "long_range_temporal"
        long_cfg = self.config.qa_long_range if long_range else None
        if long_range:
            assert long_cfg is not None
            cfg = replace(
                cfg,
                screening_frames_per_video=long_cfg.screening_frames_per_video,
                representatives_per_video=long_cfg.representatives_per_video,
                max_candidates_per_video=long_cfg.max_candidates_per_video,
            )
        eligible_criteria = [
            criterion
            for criterion in plan.criteria
            if criterion.required
            and criterion.criterion_type != "context"
            and criterion.criterion_type != "temporal"
            and criterion.description.strip()
        ][: cfg.max_retrieval_criteria]
        retrieval_queries = [
            {"query_id": "main", "text": plan.main_visual_query_en},
            *[
                {
                    "query_id": criterion.criterion_id,
                    "text": criterion.description,
                    "criterion_type": criterion.criterion_type,
                    "evidence_scope": criterion.evidence_scope,
                }
                for criterion in eligible_criteria
            ],
        ]
        existing_texts = {
            str(item["text"]).strip().casefold() for item in retrieval_queries
        }
        if long_range and long_cfg.include_event_queries:
            for event in plan.events[: long_cfg.max_event_queries]:
                text = event.visual_query_en.strip()
                if not text or text.casefold() in existing_texts:
                    continue
                existing_texts.add(text.casefold())
                retrieval_queries.append(
                    {
                        "query_id": f"qa_event_{event.event_id}",
                        "text": text,
                        "criterion_type": "visual",
                        "evidence_scope": "video",
                    }
                )
        if (
            long_range
            and long_cfg.answer_locator_query_enabled
            and plan.target_object.strip()
        ):
            locator = (
                "A clear close view of "
                f"{plan.target_object.strip()}, with the requested answer visibly "
                "readable or directly identifiable."
            )
            if locator.casefold() not in existing_texts:
                retrieval_queries.append(
                    {
                        "query_id": "qa_answer_locator",
                        "text": locator,
                        "criterion_type": "exact_text"
                        if plan.answer_expected_visible
                        else "visual",
                        "evidence_scope": "video",
                    }
                )
        # Keep at least one independent evidence query when the planner supplied
        # no usable criterion. Support queries are still derived from the original.
        if len(retrieval_queries) == 1:
            retrieval_queries.extend(
                {
                    "query_id": f"support_{index}",
                    "text": text,
                    "criterion_type": "visual",
                    "evidence_scope": "frame",
                }
                for index, text in enumerate(
                    plan.support_queries_en[: cfg.max_retrieval_criteria], start=1
                )
                if text.strip()
            )
        embeddings = self.encoder.encode(
            [str(item["text"]) for item in retrieval_queries]
        )
        qa_retriever = FeatureZipRetriever(
            feature_zip=self.config.data.feature_zip,
            top_per_video=cfg.top_frames_per_criterion_per_video,
            shortlist_size=cfg.retrieval_shortlist_size,
            max_candidates_per_video=cfg.max_candidates_per_video,
        )
        evidence_by_video, video_ranking = qa_retriever.retrieve_video_coverage(
            embeddings,
            [str(item["query_id"]) for item in retrieval_queries],
            top_frames_per_query=cfg.top_frames_per_criterion_per_video,
            coverage_rank_cutoff=cfg.coverage_rank_cutoff,
            video_limit=cfg.retrieval_shortlist_size,
            coverage_weight=cfg.coverage_weight,
            evidence_score_weight=cfg.evidence_score_weight,
            main_query_weight=cfg.main_query_weight,
            preserve_main_query_videos=(
                long_cfg.preserve_main_query_videos
                if long_range and long_cfg is not None
                else 0
            ),
        )
        pool_size = cfg.candidate_videos * cfg.max_rounds
        screening_summary: dict[str, Any] = {"enabled": False}
        ranked_ids = [str(row["video_id"]) for row in video_ranking]
        selected_video_ids = ranked_ids[:pool_size]
        if cfg.screening_enabled:
            screening_ids = ranked_ids[: cfg.screening_videos]
            screening_candidates: list[Candidate] = []
            for video_id in screening_ids:
                sparse = self._qa_round_robin_candidates(
                    evidence_by_video.get(video_id, []),
                    cfg.screening_frames_per_video,
                )
                screening_candidates.extend(self._materialize(sparse))
            if screening_candidates:
                screening_dir = run_dir / f"{artifact_prefix}_screening"
                screening_sheets = self._candidate_sheets(
                    screening_candidates, screening_dir, "evidence"
                )
                write_json(
                    screening_dir / "candidates.json",
                    [candidate.to_dict() for candidate in screening_candidates],
                )
                screening_decision = self._client().screen_qa_videos(
                    plan.original_query,
                    plan,
                    screening_candidates,
                    screening_sheets,
                    pool_size,
                )
                write_json(
                    screening_dir / "vlm_video_screening.json", screening_decision
                )
                available = {candidate.video_id for candidate in screening_candidates}
                screened_ids: list[str] = []
                if long_range:
                    # A full-story query can find the correct video even when one
                    # weak event loses the coverage score. Never let screening eject
                    # the strongest full-query candidates before timeline review.
                    main_ranked = sorted(
                        video_ranking,
                        key=lambda row: int(row.get("main_query_rank", 10**9)),
                    )
                    for row in main_ranked[: long_cfg.preserve_main_query_videos]:
                        video_id = str(row.get("video_id", ""))
                        if video_id in available and video_id not in screened_ids:
                            screened_ids.append(video_id)
                for row in screening_decision.get("selected", []):
                    video_id = str(row.get("video_id", ""))
                    if video_id in available and video_id not in screened_ids:
                        screened_ids.append(video_id)
                    if len(screened_ids) >= pool_size:
                        break
                for video_id in ranked_ids:
                    if video_id not in screened_ids:
                        screened_ids.append(video_id)
                    if len(screened_ids) >= pool_size:
                        break
                selected_video_ids = screened_ids
                screening_summary = {
                    "enabled": True,
                    "screened_video_ids": screening_ids,
                    "selected_video_ids": selected_video_ids,
                    "reason": screening_decision.get("reason", ""),
                    "contact_sheets": [str(path) for path in screening_sheets],
                }
        rows_by_video = {str(row["video_id"]): row for row in video_ranking}
        selected_rows = [
            rows_by_video[video_id]
            for video_id in selected_video_ids
            if video_id in rows_by_video
        ]
        video_ids = [str(row["video_id"]) for row in selected_rows]
        score_by_video = {
            str(row["video_id"]): float(row["joint_score"])
            for row in selected_rows
        }
        all_candidates: list[Candidate] = []
        for video_id in video_ids:
            try:
                total = len(self.mapping.rows(video_id))
            except (KeyError, ValueError):
                continue
            uniform = self._materialize_video_ordinals(
                video_id,
                self._uniform_ordinals(total, cfg.representatives_per_video),
                score_by_video[video_id],
                source_round=401,
            )
            by_id = {candidate.candidate_id: candidate for candidate in uniform}
            retrieved = self._materialize(evidence_by_video.get(video_id, []))
            for candidate in retrieved:
                current = by_id.get(candidate.candidate_id)
                if current is None or candidate.retrieval_score > current.retrieval_score:
                    by_id[candidate.candidate_id] = candidate
            selected = sorted(by_id.values(), key=lambda candidate: candidate.ordinal)
            if len(selected) > cfg.max_candidates_per_video:
                # Preserve timeline coverage, then round-robin CLIP evidence so one
                # visually easy criterion cannot occupy every remaining slot.
                uniform_ids = {candidate.candidate_id for candidate in uniform}
                preserved = [
                    candidate for candidate in selected if candidate.candidate_id in uniform_ids
                ]
                remaining = max(0, cfg.max_candidates_per_video - len(preserved))
                extras = self._qa_round_robin_candidates(
                    [
                        candidate
                        for candidate in selected
                        if candidate.candidate_id not in uniform_ids
                    ],
                    remaining,
                )
                selected = sorted(preserved + extras, key=lambda candidate: candidate.ordinal)
            all_candidates.extend(selected)
        materialized_video_ids = {candidate.video_id for candidate in all_candidates}
        video_ids = [
            video_id for video_id in video_ids if video_id in materialized_video_ids
        ]
        write_json(
            run_dir / f"{artifact_prefix}_retrieval.json",
            {
                "strategy": "criterion_video_coverage",
                "retrieval_queries": retrieval_queries,
                "candidate_videos": video_ids,
                "video_ranking": video_ranking,
                "screening": screening_summary,
                "representatives": [candidate.to_dict() for candidate in all_candidates],
            },
        )
        return all_candidates, video_ids, video_ranking

    def _normalize_qa_result(
        self,
        plan: QueryPlan,
        row: dict[str, Any],
        by_id: dict[str, Candidate],
    ) -> dict[str, Any]:
        video_id = str(row.get("video_id", ""))
        evidence: list[dict[str, Any]] = []
        evidence_by_id: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for candidate_id in row.get("evidence_candidate_ids", []):
            candidate = by_id.get(str(candidate_id))
            if candidate is None or candidate.video_id != video_id or candidate_id in seen:
                continue
            seen.add(str(candidate_id))
            frame = {
                    "candidate_id": candidate.candidate_id,
                    "video_id": candidate.video_id,
                    "frame_idx": candidate.frame_idx,
                    "keyframe_ordinal": candidate.ordinal,
                    "pts_time": candidate.pts_time,
                    "fps": candidate.fps,
                    "keyframe_path": str(candidate.image_path),
                }
            evidence.append(frame)
            evidence_by_id[candidate.candidate_id] = frame
        answer_evidence: list[dict[str, Any]] = []
        for candidate_id in row.get("answer_evidence_candidate_ids", []):
            candidate_id = str(candidate_id)
            frame = evidence_by_id.get(candidate_id)
            if frame is None:
                candidate = by_id.get(candidate_id)
                if candidate is None or candidate.video_id != video_id:
                    continue
                frame = {
                    "candidate_id": candidate.candidate_id,
                    "video_id": candidate.video_id,
                    "frame_idx": candidate.frame_idx,
                    "keyframe_ordinal": candidate.ordinal,
                    "pts_time": candidate.pts_time,
                    "fps": candidate.fps,
                    "keyframe_path": str(candidate.image_path),
                }
                if candidate_id not in seen:
                    seen.add(candidate_id)
                    evidence.append(frame)
                    evidence_by_id[candidate_id] = frame
            if frame not in answer_evidence:
                answer_evidence.append(frame)
        failures: list[dict[str, Any]] = []
        for criterion in plan.criteria:
            if not criterion.required or criterion.criterion_type == "context":
                continue
            if criterion.classification_confidence < (
                self.config.verification.classification_confidence_threshold
            ):
                continue
            observation = next(
                (
                    item
                    for item in row.get("criterion_results", [])
                    if item.get("criterion_id") == criterion.criterion_id
                ),
                None,
            )
            if not (
                observation
                and observation.get("satisfied")
                and float(observation.get("confidence", 0.0))
                >= self.config.verification.criterion_confidence_threshold
            ):
                failures.append(
                    {
                        "criterion_id": criterion.criterion_id,
                        "description": criterion.description,
                        "evidence_scope": criterion.evidence_scope,
                        "evidence": observation,
                    }
                )
        answer = row.get("answer")
        if plan.answer_expected_visible and not answer:
            failures.append(
                {
                    "criterion_id": "qa_answer",
                    "description": "The requested answer must be visibly supported.",
                    "evidence_scope": "frame",
                    "evidence": None,
                }
            )
        if plan.answer_expected_visible and answer and not answer_evidence:
            failures.append(
                {
                    "criterion_id": "qa_answer_evidence",
                    "description": "The visible answer needs its own cited frame.",
                    "evidence_scope": "frame",
                    "evidence": None,
                }
            )
        verified = (
            bool(row.get("verified"))
            and bool(evidence)
            and not failures
            and float(row.get("confidence", 0.0))
            >= self.config.qa.minimum_confidence
        )
        return {
            "video_id": video_id,
            "answer": answer,
            "answer_source": row.get("answer_source", "not_visible"),
            "verified": verified,
            "model_verified": bool(row.get("verified")),
            "confidence": float(row.get("confidence", 0.0)),
            "target_object": plan.target_object,
            "target_object_visible": bool(row.get("target_object_visible")),
            "count_status": row.get("count_status", "not_applicable"),
            "evidence_frames": evidence,
            "answer_evidence_frames": answer_evidence,
            "criterion_results": row.get("criterion_results", []),
            "hard_gate_passed": not failures,
            "hard_gate_failures": failures,
            "missing": row.get("missing", []),
            "reason": row.get("reason", ""),
        }

    def _run_qa(
        self,
        query: str,
        plan: QueryPlan,
        run_dir: Path,
        qa_mode: str = "single_frame",
        qa_mode_source: str = "default",
    ) -> dict[str, Any]:
        if not self.config.qa.enabled:
            raise RuntimeError("task_type='qa' requires qa.enabled=true")
        candidates, ranked_video_ids, video_ranking = self._prepare_qa_candidates(
            plan, run_dir, qa_mode=qa_mode
        )
        if not candidates:
            result = {
                "task_type": "qa",
                "qa_mode": qa_mode,
                "qa_mode_source": qa_mode_source,
                "strategy": (
                    "qa_long_range_temporal"
                    if qa_mode == "long_range_temporal"
                    else "qa_criterion_coverage"
                ),
                "verified": False,
                "query": query,
                "question": plan.qa_question,
                "target_object": plan.target_object,
                "results": [],
                "reason": "No QA video candidates could be materialized.",
                "run_dir": str(run_dir),
            }
            write_json(run_dir / "result.json", result)
            return result
        qa_dir = run_dir / "qa_evidence"
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        attempted_video_ids: list[str] = []
        attempted_results: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        last_reason = ""
        cfg = self.config.qa
        for round_index in range(cfg.max_rounds):
            start = round_index * cfg.candidate_videos
            round_video_ids = ranked_video_ids[start : start + cfg.candidate_videos]
            if not round_video_ids:
                break
            round_set = set(round_video_ids)
            round_candidates = [
                candidate for candidate in candidates if candidate.video_id in round_set
            ]
            if not round_candidates:
                continue
            attempted_video_ids.extend(round_video_ids)
            round_dir = qa_dir / f"round_{round_index + 1:02d}"
            sheets = self._candidate_sheets(
                round_candidates, round_dir, "representatives"
            )
            write_json(
                round_dir / "candidates.json",
                [candidate.to_dict() for candidate in round_candidates],
            )
            decision = self._client().select_qa_evidence(
                query, plan, round_candidates, sheets, cfg.result_count
            )
            write_json(round_dir / "vlm_qa_evidence.json", decision)
            last_reason = str(decision.get("reason", ""))
            round_results: list[dict[str, Any]] = []
            seen_round_videos: set[str] = set()
            for row in decision.get("results", []):
                if not isinstance(row, dict):
                    continue
                item = self._normalize_qa_result(plan, row, by_id)
                video_id = item["video_id"]
                if video_id not in round_set or video_id in seen_round_videos:
                    continue
                item["qa_round"] = round_index + 1
                seen_round_videos.add(video_id)
                round_results.append(item)
            attempted_results.extend(round_results)
            round_verified = any(item["verified"] for item in round_results)
            attempts.append(
                {
                    "round": round_index + 1,
                    "video_ids": round_video_ids,
                    "verified": round_verified,
                    "reason": last_reason,
                    "contact_sheets": [str(path) for path in sheets],
                }
            )
            if round_verified or not cfg.fallback_on_unverified:
                break

        # Keep the strongest assessment for every attempted video.
        best_by_video: dict[str, dict[str, Any]] = {}
        for item in attempted_results:
            current = best_by_video.get(item["video_id"])
            if current is None or (
                bool(item["verified"]), float(item["confidence"])
            ) > (bool(current["verified"]), float(current["confidence"])):
                best_by_video[item["video_id"]] = item
        normalized = list(best_by_video.values())

        # Preserve reviewability if a model response omitted or misspelled a video.
        seen_videos = set(best_by_video)
        for video_id in attempted_video_ids:
            if video_id in seen_videos:
                continue
            fallback_frame = next(
                (candidate for candidate in candidates if candidate.video_id == video_id),
                None,
            )
            if fallback_frame is None:
                continue
            seen_videos.add(video_id)
            normalized.append(
                self._qa_fallback_result(
                    plan,
                    fallback_frame,
                    "The VLM did not return a valid QA evidence bundle for this video.",
                )
            )
        normalized.sort(
            key=lambda item: (bool(item["verified"]), float(item["confidence"])),
            reverse=True,
        )
        normalized = normalized[: cfg.result_count]
        # The production collection has far more than five videos.  This final
        # safeguard only covers incomplete/corrupt feature collections: keep the
        # requested five review slots, even if distinct videos are unavailable.
        if len(normalized) < self.config.qa.result_count:
            for candidate in candidates:
                if len(normalized) >= self.config.qa.result_count:
                    break
                normalized.append(
                    self._qa_fallback_result(
                        plan,
                        candidate,
                        "Fewer than five distinct video candidates were available.",
                    )
                )
        normalized.sort(
            key=lambda item: (bool(item["verified"]), float(item["confidence"])),
            reverse=True,
        )
        keyframes = [
            {"qa_rank": index, **frame}
            for index, item in enumerate(normalized, start=1)
            for frame in item["evidence_frames"]
        ]
        result = {
            "task_type": "qa",
            "qa_mode": qa_mode,
            "qa_mode_source": qa_mode_source,
            "strategy": (
                "qa_long_range_temporal"
                if qa_mode == "long_range_temporal"
                else "qa_criterion_coverage"
            ),
            "verified": bool(normalized and normalized[0]["verified"]),
            "query": query,
            "question": plan.qa_question,
            "target_object": plan.target_object,
            "candidate_videos": attempted_video_ids,
            "qa_attempts": attempts,
            "video_ranking": video_ranking,
            "results": normalized,
            "keyframes": keyframes,
            "reason": last_reason,
            "run_dir": str(run_dir),
        }
        write_json(run_dir / "result.json", result)
        self._write_result_csv(run_dir / "result.csv", result)
        return result

    @staticmethod
    def _qa_fallback_result(
        plan: QueryPlan, candidate: Candidate, reason: str
    ) -> dict[str, Any]:
        return {
            "video_id": candidate.video_id,
            "answer": None,
            "answer_source": "not_visible",
            "verified": False,
            "model_verified": False,
            "confidence": 0.0,
            "target_object": plan.target_object,
            "target_object_visible": False,
            "count_status": "not_verified",
            "evidence_frames": [
                {
                    "candidate_id": candidate.candidate_id,
                    "video_id": candidate.video_id,
                    "frame_idx": candidate.frame_idx,
                    "keyframe_ordinal": candidate.ordinal,
                    "pts_time": candidate.pts_time,
                    "fps": candidate.fps,
                    "keyframe_path": str(candidate.image_path),
                }
            ],
            "answer_evidence_frames": [],
            "criterion_results": [],
            "hard_gate_passed": False,
            "hard_gate_failures": [],
            "missing": [reason],
            "reason": "Fallback review candidate only; not verified.",
        }

    @staticmethod
    def _requires_temporal_kis(plan: QueryPlan) -> bool:
        if plan.needs_motion or (plan.sequence_required and bool(plan.events)):
            return True
        return any(
            criterion.required
            and criterion.criterion_type != "context"
            and (
                criterion.criterion_type == "temporal"
                or criterion.evidence_scope in {"window", "video"}
            )
            for criterion in plan.criteria
        )

    @staticmethod
    def _flatten_temporal_kis_plan(plan: QueryPlan) -> QueryPlan:
        """Reuse event criteria when an explicitly selected KIS was planned as TRAKE."""
        criteria = list(plan.criteria)
        seen = {criterion.criterion_id for criterion in criteria}
        for event in plan.events:
            for criterion in event.criteria:
                if criterion.criterion_id not in seen:
                    seen.add(criterion.criterion_id)
                    criteria.append(criterion)
        return replace(
            plan,
            criteria=criteria,
            task_type="kis",
            needs_motion=True,
        )

    def _normalize_temporal_kis_decision(
        self,
        plan: QueryPlan,
        candidates: list[Candidate],
        decision: dict[str, Any],
    ) -> tuple[Candidate | None, dict[str, Any]]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        video_id = str(decision.get("video_id") or "")
        selected_id = str(decision.get("selected_candidate_id") or "")
        selected = by_id.get(selected_id)
        failures: list[dict[str, Any]] = []
        if selected is None or not video_id or selected.video_id != video_id:
            selected = next(
                (
                    by_id.get(str(row.get("candidate_id", "")))
                    for row in decision.get("ranked_candidates", [])
                    if isinstance(row, dict)
                    and by_id.get(str(row.get("candidate_id", ""))) is not None
                    and by_id[str(row.get("candidate_id", ""))].video_id == video_id
                ),
                None,
            )
            selected_id = selected.candidate_id if selected is not None else ""
            failures.append(
                {
                    "criterion_id": "representative_frame",
                    "description": "A representative frame from the selected video is required.",
                    "evidence_scope": "frame",
                }
            )
            selected = None

        raw_results = [
            item
            for item in decision.get("criterion_results", [])
            if isinstance(item, dict)
        ]
        results_by_id = {
            str(item.get("criterion_id", "")): item for item in raw_results
        }
        evidence_union: list[str] = []
        normalized_criteria: list[dict[str, Any]] = []
        for criterion in plan.criteria:
            observation = results_by_id.get(criterion.criterion_id)
            valid_ids: list[str] = []
            if observation:
                for candidate_id in observation.get("evidence_candidate_ids", []):
                    candidate = by_id.get(str(candidate_id))
                    if (
                        candidate is not None
                        and candidate.video_id == video_id
                        and candidate.candidate_id not in valid_ids
                    ):
                        valid_ids.append(candidate.candidate_id)
                        if candidate.candidate_id not in evidence_union:
                            evidence_union.append(candidate.candidate_id)
                observation = {**observation, "evidence_candidate_ids": valid_ids}
                normalized_criteria.append(observation)

            if (
                not criterion.required
                or criterion.criterion_type == "context"
                or criterion.classification_confidence
                < self.config.verification.classification_confidence_threshold
            ):
                continue
            passed = bool(observation and observation.get("satisfied")) and (
                float(observation.get("confidence", 0.0))
                >= self.config.verification.criterion_confidence_threshold
            )
            if passed and not valid_ids:
                passed = False
            if passed and criterion.criterion_type == "temporal":
                times = [
                    float(by_id[candidate_id].pts_time or 0.0)
                    for candidate_id in valid_ids
                ]
                passed = len(times) >= 2 and all(
                    left < right for left, right in zip(times, times[1:])
                )
            if not passed:
                failures.append(
                    {
                        "criterion_id": criterion.criterion_id,
                        "criterion_type": criterion.criterion_type,
                        "description": criterion.description,
                        "evidence_scope": criterion.evidence_scope,
                        "evidence": observation,
                    }
                )

        if selected is not None and evidence_union and selected_id not in evidence_union:
            failures.append(
                {
                    "criterion_id": "representative_frame",
                    "description": "The chosen frame must depict a cited requested moment.",
                    "evidence_scope": "frame",
                }
            )

        confidence = float(decision.get("confidence", 0.0))
        model_verified = bool(decision.get("verified"))
        video_failures = [
            failure
            for failure in failures
            if failure.get("criterion_id") != "representative_frame"
        ]
        frame_failures = [
            failure
            for failure in failures
            if failure.get("criterion_id") == "representative_frame"
        ]
        video_verified = bool(video_id) and not video_failures
        frame_verified = selected is not None and not frame_failures
        verified = (
            model_verified
            and video_verified
            and frame_verified
            and confidence >= self.config.temporal_kis.minimum_confidence
        )

        ranked_ids: list[str] = []
        rank_rows: dict[str, dict[str, Any]] = {}
        for row in decision.get("ranked_candidates", []):
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("candidate_id", ""))
            candidate = by_id.get(candidate_id)
            if candidate is None or candidate.video_id != video_id:
                continue
            if candidate_id not in ranked_ids:
                ranked_ids.append(candidate_id)
                rank_rows[candidate_id] = row
        ordered_ids: list[str] = []
        if selected is not None:
            ordered_ids.append(selected.candidate_id)
        for candidate_id in ranked_ids + evidence_union:
            if candidate_id not in ordered_ids:
                ordered_ids.append(candidate_id)
        extras = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.video_id == video_id
                and candidate.candidate_id not in ordered_ids
            ),
            key=lambda candidate: candidate.retrieval_score,
            reverse=True,
        )
        ordered_ids.extend(candidate.candidate_id for candidate in extras)
        limit = self.config.openai.final_candidate_count
        ranked_candidates: list[dict[str, Any]] = []
        for candidate_id in ordered_ids[:limit]:
            candidate = by_id[candidate_id]
            row = rank_rows.get(candidate_id, {})
            ranked_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "video_id": candidate.video_id,
                    "frame_idx": candidate.frame_idx,
                    "keyframe_ordinal": candidate.ordinal,
                    "pts_time": candidate.pts_time,
                    "fps": candidate.fps,
                    "verified": verified and candidate_id == selected_id,
                    "confidence": confidence
                    if candidate_id == selected_id
                    else float(row.get("confidence", 0.0)),
                    "reason": decision.get("reason", "")
                    if candidate_id == selected_id
                    else row.get("reason", "Supporting temporal evidence."),
                    "keyframe_path": str(candidate.image_path),
                }
            )
        evidence_frames = [
            {
                "candidate_id": candidate_id,
                "video_id": by_id[candidate_id].video_id,
                "frame_idx": by_id[candidate_id].frame_idx,
                "keyframe_ordinal": by_id[candidate_id].ordinal,
                "pts_time": by_id[candidate_id].pts_time,
                "fps": by_id[candidate_id].fps,
                "keyframe_path": str(by_id[candidate_id].image_path),
            }
            for candidate_id in evidence_union
        ]
        normalized = {
            **decision,
            "verified": verified,
            "model_verified": model_verified,
            "video_id": video_id or None,
            "selected_candidate_id": selected.candidate_id if selected is not None else None,
            "confidence": confidence,
            "criterion_results": normalized_criteria,
            "hard_gate_passed": not failures,
            "hard_gate_failures": failures,
            "video_verified": video_verified,
            "frame_verified": frame_verified,
            "evidence_frames": evidence_frames,
            "ranked_candidates": ranked_candidates,
        }
        return selected, normalized

    def _run_temporal_kis(
        self, query: str, plan: QueryPlan, run_dir: Path
    ) -> dict[str, Any]:
        plan = self._flatten_temporal_kis_plan(plan)
        write_json(run_dir / "temporal_kis_plan.json", plan.to_dict())
        cfg = self.config.temporal_kis
        candidates, ranked_video_ids, video_ranking = self._prepare_qa_candidates(
            plan,
            run_dir,
            coverage_config=cfg,
            artifact_prefix="temporal_kis",
        )
        attempts: list[dict[str, Any]] = []
        best_selected: Candidate | None = None
        best_decision: dict[str, Any] | None = None
        attempted_video_ids: list[str] = []
        for round_index in range(cfg.max_rounds):
            start = round_index * cfg.candidate_videos
            round_video_ids = ranked_video_ids[start : start + cfg.candidate_videos]
            if not round_video_ids:
                break
            round_set = set(round_video_ids)
            round_candidates = [
                candidate for candidate in candidates if candidate.video_id in round_set
            ]
            if not round_candidates:
                continue
            attempted_video_ids.extend(round_video_ids)
            round_dir = run_dir / "temporal_kis_evidence" / f"round_{round_index + 1:02d}"
            sheets = self._candidate_sheets(
                round_candidates, round_dir, "representatives"
            )
            write_json(
                round_dir / "candidates.json",
                [candidate.to_dict() for candidate in round_candidates],
            )
            raw_decision = self._client().select_temporal_kis(
                query,
                plan,
                round_candidates,
                sheets,
                self.config.openai.final_candidate_count,
            )
            selected, decision = self._normalize_temporal_kis_decision(
                plan, round_candidates, raw_decision
            )
            write_json(round_dir / "vlm_temporal_kis.json", decision)
            attempts.append(
                {
                    "round": round_index + 1,
                    "video_ids": round_video_ids,
                    "selected_video_id": decision.get("video_id"),
                    "verified": decision.get("verified", False),
                    "confidence": decision.get("confidence", 0.0),
                    "reason": decision.get("reason", ""),
                    "contact_sheets": [str(path) for path in sheets],
                }
            )
            if best_decision is None or (
                bool(decision.get("verified")),
                selected is not None,
                float(decision.get("confidence", 0.0)),
            ) > (
                bool(best_decision.get("verified")),
                best_selected is not None,
                float(best_decision.get("confidence", 0.0)),
            ):
                best_selected = selected
                best_decision = decision
            if decision.get("verified") or not cfg.fallback_on_unverified:
                break

        if best_decision is None:
            result = {
                "verified": False,
                "task_type": "kis",
                "kis_mode": "temporal",
                "strategy": "temporal_kis_unverified",
                "query": query,
                "candidate_videos": attempted_video_ids,
                "temporal_kis_attempts": attempts,
                "video_ranking": video_ranking,
                "keyframes": [],
                "reason": "No temporal KIS evidence could be evaluated.",
                "run_dir": str(run_dir),
            }
        elif best_selected is not None:
            result = self._result_from_selection(
                query,
                run_dir,
                best_selected,
                best_decision,
                "temporal_kis",
            )
            result.update(
                {
                    "task_type": "kis",
                    "kis_mode": "temporal",
                    "video_verified": best_decision.get("video_verified", False),
                    "frame_verified": best_decision.get("frame_verified", False),
                    "evidence_frames": best_decision.get("evidence_frames", []),
                    "candidate_videos": attempted_video_ids,
                    "temporal_kis_attempts": attempts,
                    "video_ranking": video_ranking,
                }
            )
        else:
            result = {
                "verified": False,
                "task_type": "kis",
                "kis_mode": "temporal",
                "strategy": "temporal_kis_unverified",
                "query": query,
                "video_id": best_decision.get("video_id"),
                "confidence": best_decision.get("confidence", 0.0),
                "reason": best_decision.get("reason", ""),
                "missing": best_decision.get("missing", []),
                "criterion_results": best_decision.get("criterion_results", []),
                "hard_gate_passed": best_decision.get("hard_gate_passed", False),
                "hard_gate_failures": best_decision.get("hard_gate_failures", []),
                "evidence_frames": best_decision.get("evidence_frames", []),
                "keyframes": best_decision.get("ranked_candidates", []),
                "candidate_videos": attempted_video_ids,
                "temporal_kis_attempts": attempts,
                "video_ranking": video_ranking,
                "run_dir": str(run_dir),
            }
        write_json(run_dir / "result.json", result)
        self._write_result_csv(run_dir / "result.csv", result)
        return result

    @staticmethod
    def _write_result_csv(path: Path, result: dict[str, Any]) -> None:
        fields = [
            "event_id",
            "video_id",
            "frame_idx",
            "keyframe_ordinal",
            "pts_time",
            "confidence",
            "verified",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            rows = result.get("keyframes") or [result]
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})

    @staticmethod
    def _result_from_selection(
        query: str,
        run_dir: Path,
        selected: Candidate,
        decision: dict[str, Any],
        strategy: str,
    ) -> dict[str, Any]:
        return {
            "verified": bool(decision.get("verified")),
            "strategy": strategy,
            "query": query,
            "video_id": selected.video_id,
            "frame_idx": selected.frame_idx,
            "keyframe_ordinal": selected.ordinal,
            "pts_time": selected.pts_time,
            "fps": selected.fps,
            "confidence": float(decision.get("confidence", 0.0)),
            "reason": decision.get("reason", ""),
            "satisfied": decision.get("satisfied", []),
            "missing": decision.get("missing", []),
            "criterion_results": decision.get("criterion_results", []),
            "hard_gate_passed": decision.get("hard_gate_passed"),
            "hard_gate_failures": decision.get("hard_gate_failures", []),
            "keyframe_path": str(selected.image_path),
            "keyframes": decision.get("ranked_candidates", []),
            "run_dir": str(run_dir),
        }

    @staticmethod
    def _normalize_task_type(task_type: str | None) -> str:
        value = str(task_type or "auto").strip().lower()
        aliases = {
            "auto": "auto",
            "kis": "kis",
            "trake": "trake",
            "qa": "qa",
        }
        if value not in aliases:
            raise ValueError("task_type must be one of: auto, kis, trake, qa")
        return aliases[value]

    @staticmethod
    def _normalize_qa_mode(qa_mode: str | None) -> str:
        value = str(qa_mode or "auto").strip().lower()
        if value == "not_applicable":
            return "auto"
        allowed = {"auto", "single_frame", "short_window", "long_range_temporal"}
        if value not in allowed:
            raise ValueError(
                "qa_mode must be one of: auto, single_frame, short_window, "
                "long_range_temporal"
            )
        return value

    @staticmethod
    def _infer_qa_mode(plan: QueryPlan) -> str:
        """Defensive local fallback when a planner leaves qa_mode undecided."""
        if len(plan.events) >= 2:
            return "long_range_temporal"
        required = [
            criterion
            for criterion in plan.criteria
            if criterion.required and criterion.criterion_type != "context"
        ]
        video_scoped = sum(
            criterion.evidence_scope == "video" for criterion in required
        )
        temporal_markers = (
            "sau đó",
            "chuyển sang cảnh",
            "tiếp theo",
            "về sau",
            "then",
            "afterward",
            "later in the video",
        )
        lowered = plan.original_query.casefold()
        if (
            plan.sequence_required
            and (video_scoped >= 2 or len(required) >= 3)
        ) or (
            len(required) >= 2
            and any(marker in lowered for marker in temporal_markers)
            and plan.answer_expected_visible
        ):
            return "long_range_temporal"
        if plan.needs_motion or any(
            criterion.evidence_scope in {"window", "video"}
            or criterion.criterion_type == "temporal"
            for criterion in required
        ):
            return "short_window"
        return "single_frame"

    def run(
        self,
        query: str,
        retrieval_only: bool | None = None,
        task_type: str = "auto",
        qa_mode: str = "auto",
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        requested_task_type = self._normalize_task_type(task_type)
        requested_qa_mode = self._normalize_qa_mode(qa_mode)
        if retrieval_only is None:
            retrieval_only = self.config.runtime.retrieval_only

        run_dir = make_run_dir(self.config.data.output_dir, query)
        plan = self._prepare_plan(query, retrieval_only)
        write_json(run_dir / "query_plan.json", plan.to_dict())

        if retrieval_only:
            embeddings = self.encoder.encode([plan.main_visual_query_en])
            candidates = self.retriever.retrieve(
                embeddings,
                main_weight=1.0,
                support_weight=0.0,
            )
            candidates = self._materialize(candidates)
            sheets = self._candidate_sheets(candidates, run_dir, "retrieval_only")
            result = {
                "mode": "retrieval_only",
                "task_type": requested_task_type,
                "run_dir": str(run_dir),
                "contact_sheets": [str(path) for path in sheets],
                "candidates": [item.to_dict() for item in candidates],
            }
            write_json(run_dir / "result.json", result)
            return result

        config_task_type = self._normalize_task_type(
            self.config.runtime.task_type_override
        )
        if plan.task_type == "qa":
            inferred_task_type = "qa"
        elif plan.task_type == "trake" or (
            plan.sequence_required and len(plan.events) >= 2
        ):
            inferred_task_type = "trake"
        else:
            inferred_task_type = "kis"
        if requested_task_type != "auto":
            effective_task_type = requested_task_type
        elif config_task_type != "auto":
            effective_task_type = config_task_type
        else:
            effective_task_type = inferred_task_type
        if effective_task_type == "qa":
            config_qa_mode = self._normalize_qa_mode(
                self.config.runtime.qa_mode_override
            )
            planned_qa_mode = self._normalize_qa_mode(plan.qa_mode)
            if requested_qa_mode != "auto":
                effective_qa_mode = requested_qa_mode
                qa_mode_source = "run_argument"
            elif config_qa_mode != "auto":
                effective_qa_mode = config_qa_mode
                qa_mode_source = "config_override"
            elif self.config.qa_long_range.auto_detect:
                effective_qa_mode = (
                    planned_qa_mode
                    if planned_qa_mode != "auto"
                    else self._infer_qa_mode(plan)
                )
                qa_mode_source = (
                    "planner" if planned_qa_mode != "auto" else "local_heuristic"
                )
            else:
                effective_qa_mode = "single_frame"
                qa_mode_source = "auto_detection_disabled"
            if (
                effective_qa_mode == "long_range_temporal"
                and not self.config.qa_long_range.enabled
            ):
                effective_qa_mode = "short_window"
                qa_mode_source += "_long_range_disabled"
            qa_plan = replace(plan, task_type="qa", qa_mode=effective_qa_mode)
            write_json(run_dir / "effective_query_plan.json", qa_plan.to_dict())
            return self._run_qa(
                query,
                qa_plan,
                run_dir,
                effective_qa_mode,
                qa_mode_source,
            )
        if effective_task_type == "trake":
            if not self.config.multi_event.enabled:
                raise RuntimeError(
                    "task_type='trake' requires multi_event.enabled=true"
                )
            if len(plan.events) < 2:
                raise RuntimeError(
                    "TRAKE requires at least two planned events. Use explicit labels "
                    "such as E1:, E2:, ... in the query."
                )
            return self._run_multi_event(query, plan, run_dir)
        if (
            effective_task_type == "kis"
            and self.config.temporal_kis.enabled
            and self._requires_temporal_kis(plan)
        ):
            return self._run_temporal_kis(query, plan, run_dir)

        finalists, rankings, candidate_history = self._retrieve_and_compare(query, plan, run_dir)
        primary_selected, primary_decision = self._attempt_final_selection(
            query, plan, finalists, run_dir
        )
        if self._is_verified(
            primary_selected,
            primary_decision,
            self.config.refinement.final_confidence_threshold,
        ):
            result = self._result_from_selection(
                query, run_dir, primary_selected, primary_decision, "primary"
            )
            result["task_type"] = "kis"
            result["kis_mode"] = "static"
            write_json(run_dir / "result.json", result)
            self._write_result_csv(run_dir / "result.csv", result)
            return result

        rescue_selected, rescue_decision, preserved_anchors, rescue_summary = (
            self._attempt_rescue_zoom(
                query, plan, rankings, candidate_history, run_dir
            )
        )
        if self._is_verified(
            rescue_selected,
            rescue_decision,
            self.config.refinement.final_confidence_threshold,
        ):
            result = self._result_from_selection(
                query, run_dir, rescue_selected, rescue_decision, "rescue_zoom"
            )
            result["task_type"] = "kis"
            result["kis_mode"] = "static"
            result["rescue_zoom"] = rescue_summary
            write_json(run_dir / "result.json", result)
            self._write_result_csv(run_dir / "result.csv", result)
            return result

        should_fallback = self.config.fallback.enabled and (
            primary_selected is None or self.config.fallback.trigger_on_unverified
        )
        fallback_summary: dict[str, Any] | None = None
        fallback_decision: dict[str, Any] | None = None
        if should_fallback:
            fallback_centers, fallback_summary = self._video_level_fallback(
                query, plan, candidate_history, run_dir, preserved_anchors
            )
            fallback_selected, fallback_decision = self._attempt_final_selection(
                query,
                plan,
                fallback_centers,
                run_dir / "fallback_video_scan",
            )
            if self._is_verified(
                fallback_selected,
                fallback_decision,
                self.config.refinement.final_confidence_threshold,
            ):
                result = self._result_from_selection(
                    query,
                    run_dir,
                    fallback_selected,
                    fallback_decision,
                    "video_level_fallback",
                )
                result["task_type"] = "kis"
                result["kis_mode"] = "static"
                result["fallback"] = fallback_summary
                result["rescue_zoom"] = rescue_summary
                result["rescue_zoom_final_decision"] = rescue_decision
                write_json(run_dir / "result.json", result)
                self._write_result_csv(run_dir / "result.csv", result)
                return result

        if primary_selected is not None:
            result = self._result_from_selection(
                query, run_dir, primary_selected, primary_decision, "primary_unverified"
            )
        else:
            result = {
                "verified": False,
                "strategy": "unverified",
                "query": query,
                "run_dir": str(run_dir),
                "reason": primary_decision.get(
                    "reason", "The VLM did not select a valid retrieval candidate."
                ),
            }
        result["verified"] = False
        result["task_type"] = "kis"
        result["kis_mode"] = "static"
        # Even without an accepted answer, expose the strongest final VLM ranking
        # so the notebook can show 5-10 reviewable alternatives.
        review_decisions = [fallback_decision, rescue_decision, primary_decision]
        review_decision = next(
            (
                decision
                for decision in review_decisions
                if decision and decision.get("ranked_candidates")
            ),
            None,
        )
        if review_decision is not None:
            result["keyframes"] = review_decision["ranked_candidates"]
        result["rankings"] = rankings
        result["primary_final_decision"] = primary_decision
        result["rescue_zoom"] = rescue_summary
        result["rescue_zoom_final_decision"] = rescue_decision
        if fallback_summary is not None:
            result["fallback"] = fallback_summary
            result["fallback_final_decision"] = fallback_decision
        write_json(run_dir / "result.json", result)
        self._write_result_csv(run_dir / "result.csv", result)
        return result
