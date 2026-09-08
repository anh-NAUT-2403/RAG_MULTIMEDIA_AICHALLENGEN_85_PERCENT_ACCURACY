from __future__ import annotations

import io
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm.auto import tqdm

from ..models import Candidate


class FeatureZipRetriever:
    """Stream precomputed per-video CLIP arrays without unpacking the ZIP."""

    def __init__(
        self,
        feature_zip: Path,
        top_per_video: int,
        shortlist_size: int,
        max_candidates_per_video: int,
    ) -> None:
        self.feature_zip = Path(feature_zip)
        self.top_per_video = int(top_per_video)
        self.shortlist_size = int(shortlist_size)
        self.max_candidates_per_video = int(max_candidates_per_video)

    @staticmethod
    def _weights(count: int, main_weight: float, support_weight: float) -> np.ndarray:
        if count < 1:
            raise ValueError("No text embeddings supplied")
        if count == 1:
            return np.array([1.0], dtype=np.float32)
        weights = np.array(
            [main_weight] + [support_weight] * (count - 1), dtype=np.float32
        )
        return weights / weights.sum()

    def retrieve(
        self,
        text_embeddings: np.ndarray,
        main_weight: float,
        support_weight: float,
        source_round: int = 1,
        exclude_ids: set[str] | None = None,
    ) -> list[Candidate]:
        exclude_ids = exclude_ids or set()
        weights = self._weights(len(text_embeddings), main_weight, support_weight)
        pooled: list[tuple[float, float, list[float], str, int]] = []

        with zipfile.ZipFile(self.feature_zip) as archive:
            members = [name for name in archive.namelist() if name.endswith(".npy")]
            for member in tqdm(members, desc="CLIP retrieval", unit="video"):
                features = np.load(io.BytesIO(archive.read(member))).astype(
                    np.float32, copy=False
                )
                if features.ndim != 2 or features.shape[0] == 0:
                    continue
                features /= np.maximum(
                    np.linalg.norm(features, axis=1, keepdims=True), 1e-12
                )
                component_scores = features @ text_embeddings.T
                combined = component_scores @ weights
                count = min(self.top_per_video, len(combined))
                indices = np.argpartition(combined, len(combined) - count)[-count:]
                video_id = PurePosixPath(member).stem
                for index in indices:
                    ordinal = int(index) + 1
                    candidate_id = f"{video_id}__K{ordinal:03d}"
                    if candidate_id in exclude_ids:
                        continue
                    pooled.append(
                        (
                            float(combined[index]),
                            float(component_scores[index, 0]),
                            [float(value) for value in component_scores[index, 1:]],
                            video_id,
                            ordinal,
                        )
                    )

        pooled.sort(key=lambda row: row[0], reverse=True)
        selected: list[Candidate] = []
        per_video: defaultdict[str, int] = defaultdict(int)
        for score, main_score, support_scores, video_id, ordinal in pooled:
            if per_video[video_id] >= self.max_candidates_per_video:
                continue
            selected.append(
                Candidate(
                    candidate_id=f"{video_id}__K{ordinal:03d}",
                    video_id=video_id,
                    ordinal=ordinal,
                    retrieval_score=score,
                    main_score=main_score,
                    support_scores=support_scores,
                    source_round=source_round,
                )
            )
            per_video[video_id] += 1
            if len(selected) >= self.shortlist_size:
                break
        return selected

    def retrieve_multi_event(
        self,
        text_embeddings: np.ndarray,
        candidate_videos: int,
        top_frames_per_event_per_video: int,
    ) -> tuple[dict[str, list[list[Candidate]]], list[dict[str, object]]]:
        """Rank videos by their weakest event, then keep top frames per event."""
        if text_embeddings.ndim != 2 or len(text_embeddings) < 2:
            raise ValueError("Multi-event retrieval requires at least two embeddings")
        per_video: dict[str, list[list[Candidate]]] = {}
        video_rows: list[dict[str, object]] = []

        with zipfile.ZipFile(self.feature_zip) as archive:
            members = [name for name in archive.namelist() if name.endswith(".npy")]
            for member in tqdm(members, desc="Multi-event CLIP", unit="video"):
                features = np.load(io.BytesIO(archive.read(member))).astype(
                    np.float32, copy=False
                )
                if features.ndim != 2 or features.shape[0] == 0:
                    continue
                features /= np.maximum(
                    np.linalg.norm(features, axis=1, keepdims=True), 1e-12
                )
                scores = features @ text_embeddings.T
                video_id = PurePosixPath(member).stem
                event_candidates: list[list[Candidate]] = []
                event_scores: list[float] = []
                for event_index in range(scores.shape[1]):
                    column = scores[:, event_index]
                    count = min(top_frames_per_event_per_video, len(column))
                    indices = np.argpartition(column, len(column) - count)[-count:]
                    ordered = sorted(indices, key=lambda index: column[index], reverse=True)
                    candidates = [
                        Candidate(
                            candidate_id=f"{video_id}__K{int(index) + 1:03d}",
                            video_id=video_id,
                            ordinal=int(index) + 1,
                            retrieval_score=float(column[index]),
                            main_score=float(column[index]),
                            source_round=200 + event_index,
                        )
                        for index in ordered
                    ]
                    event_candidates.append(candidates)
                    event_scores.append(float(column[ordered[0]]))
                # The weakest event dominates: a video strong for only one fruit
                # must not outrank a video containing the complete sequence.
                joint_score = 0.7 * min(event_scores) + 0.3 * float(
                    np.mean(event_scores)
                )
                per_video[video_id] = event_candidates
                video_rows.append(
                    {
                        "video_id": video_id,
                        "joint_score": joint_score,
                        "event_scores": event_scores,
                    }
                )

        video_rows.sort(key=lambda row: float(row["joint_score"]), reverse=True)
        selected_rows = video_rows[:candidate_videos]
        selected_ids = {str(row["video_id"]) for row in selected_rows}
        return (
            {video_id: rows for video_id, rows in per_video.items() if video_id in selected_ids},
            selected_rows,
        )

    def retrieve_video_coverage(
        self,
        text_embeddings: np.ndarray,
        query_ids: list[str],
        *,
        top_frames_per_query: int,
        coverage_rank_cutoff: int,
        video_limit: int,
        coverage_weight: float,
        evidence_score_weight: float,
        main_query_weight: float,
        source_round: int = 450,
        preserve_main_query_videos: int = 0,
    ) -> tuple[dict[str, list[Candidate]], list[dict[str, object]]]:
        """Rank videos by evidence-query coverage while streaming features once.

        Query index zero is the complete authoritative query. Remaining entries are
        independently derived visual criteria. A video therefore benefits from
        matching several different criteria, even when those matches occur in
        different keyframes.
        """
        if text_embeddings.ndim != 2 or len(text_embeddings) != len(query_ids):
            raise ValueError("query_ids must align with the text embedding rows")
        if not query_ids:
            raise ValueError("At least one QA retrieval query is required")

        per_video_candidates: dict[str, list[Candidate]] = {}
        video_ids: list[str] = []
        score_rows: list[np.ndarray] = []

        with zipfile.ZipFile(self.feature_zip) as archive:
            members = [name for name in archive.namelist() if name.endswith(".npy")]
            for member in tqdm(members, desc="QA evidence CLIP", unit="video"):
                features = np.load(io.BytesIO(archive.read(member))).astype(
                    np.float32, copy=False
                )
                if features.ndim != 2 or features.shape[0] == 0:
                    continue
                features /= np.maximum(
                    np.linalg.norm(features, axis=1, keepdims=True), 1e-12
                )
                scores = features @ text_embeddings.T
                video_id = PurePosixPath(member).stem
                maxima = np.max(scores, axis=0)
                video_ids.append(video_id)
                score_rows.append(maxima)

                by_id: dict[str, Candidate] = {}
                for query_index in range(scores.shape[1]):
                    column = scores[:, query_index]
                    count = min(top_frames_per_query, len(column))
                    indices = np.argpartition(column, len(column) - count)[-count:]
                    for index in indices:
                        ordinal = int(index) + 1
                        candidate_id = f"{video_id}__K{ordinal:03d}"
                        score = float(column[index])
                        current = by_id.get(candidate_id)
                        if current is None or score > current.retrieval_score:
                            by_id[candidate_id] = Candidate(
                                candidate_id=candidate_id,
                                video_id=video_id,
                                ordinal=ordinal,
                                retrieval_score=score,
                                main_score=float(scores[index, 0]),
                                source_round=source_round + query_index,
                            )
                per_video_candidates[video_id] = list(by_id.values())

        if not score_rows:
            return {}, []

        matrix = np.stack(score_rows).astype(np.float32, copy=False)
        video_count, query_count = matrix.shape
        ranks = np.empty((video_count, query_count), dtype=np.int32)
        for query_index in range(query_count):
            order = np.argsort(-matrix[:, query_index], kind="stable")
            ranks[order, query_index] = np.arange(video_count, dtype=np.int32)
        if video_count == 1:
            percentiles = np.ones_like(matrix, dtype=np.float32)
        else:
            percentiles = 1.0 - ranks.astype(np.float32) / float(video_count - 1)

        evidence_indices = list(range(1, query_count)) or [0]
        cutoff = min(max(1, coverage_rank_cutoff), video_count)
        rows: list[dict[str, object]] = []
        for video_index, video_id in enumerate(video_ids):
            evidence_ranks = ranks[video_index, evidence_indices]
            coverage_count = int(np.sum(evidence_ranks < cutoff))
            coverage_ratio = coverage_count / len(evidence_indices)
            evidence_percentile = float(
                np.mean(percentiles[video_index, evidence_indices])
            )
            main_percentile = float(percentiles[video_index, 0])
            joint_score = (
                coverage_weight * coverage_ratio
                + evidence_score_weight * evidence_percentile
                + main_query_weight * main_percentile
            )
            rows.append(
                {
                    "video_id": video_id,
                    "joint_score": joint_score,
                    "coverage_count": coverage_count,
                    "coverage_total": len(evidence_indices),
                    "coverage_ratio": coverage_ratio,
                    "main_query_rank": int(ranks[video_index, 0]) + 1,
                    "main_query_score": float(matrix[video_index, 0]),
                    "evidence": [
                        {
                            "query_id": query_ids[index],
                            "rank": int(ranks[video_index, index]) + 1,
                            "score": float(matrix[video_index, index]),
                            "covered": bool(ranks[video_index, index] < cutoff),
                        }
                        for index in evidence_indices
                    ],
                }
            )

        rows.sort(
            key=lambda row: (
                float(row["joint_score"]),
                int(row["coverage_count"]),
                float(row["main_query_score"]),
            ),
            reverse=True,
        )
        selected_rows = rows[:video_limit]
        if preserve_main_query_videos > 0:
            selected_ids = {str(row["video_id"]) for row in selected_rows}
            main_rows = sorted(
                rows, key=lambda row: int(row["main_query_rank"])
            )[:preserve_main_query_videos]
            for row in main_rows:
                if str(row["video_id"]) not in selected_ids:
                    selected_rows.append(row)
                    selected_ids.add(str(row["video_id"]))
        selected_ids = {str(row["video_id"]) for row in selected_rows}
        selected_candidates = {
            video_id: candidates
            for video_id, candidates in per_video_candidates.items()
            if video_id in selected_ids
        }
        return selected_candidates, selected_rows
