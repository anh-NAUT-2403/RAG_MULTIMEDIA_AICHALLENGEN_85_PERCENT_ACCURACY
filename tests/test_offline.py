from __future__ import annotations

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from keyframe_agent.config import (
    AppConfig,
    ContactSheetConfig,
    DataConfig,
    FallbackConfig,
    OpenAIConfig,
    QAConfig,
    RefinementConfig,
    RetrievalConfig,
    RescueZoomConfig,
    TemporalKISConfig,
    VerificationConfig,
)
from keyframe_agent.models import Candidate, Criterion, EventPlan, QueryPlan
from keyframe_agent.pipeline import KeyframeSearchPipeline
from keyframe_agent.retrieval.feature_retriever import FeatureZipRetriever
from keyframe_agent.storage.assets import SelectiveAssetStore
from keyframe_agent.storage.mapping import KeyframeMapping
from keyframe_agent.visual.contact_sheet import ContactSheetBuilder


class OfflinePipelineComponentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_feature_zip(self) -> Path:
        path = self.root / "features.zip"
        arrays = {
            "clip-features-32/L01_V001.npy": np.array(
                [[1.0, 0.0], [0.2, 0.8]], dtype=np.float32
            ),
            "clip-features-32/L01_V002.npy": np.array(
                [[0.8, 0.2], [0.0, 1.0]], dtype=np.float32
            ),
        }
        with zipfile.ZipFile(path, "w") as archive:
            for member, array in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, array)
                archive.writestr(member, buffer.getvalue())
        return path

    def _write_mapping_zip(self) -> Path:
        path = self.root / "mapping.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for video_id in ("L01_V001", "L01_V002"):
                buffer = io.StringIO()
                writer = csv.DictWriter(
                    buffer, fieldnames=["n", "pts_time", "fps", "frame_idx"]
                )
                writer.writeheader()
                writer.writerow({"n": 1, "pts_time": 0.0, "fps": 25, "frame_idx": 0})
                writer.writerow({"n": 2, "pts_time": 2.0, "fps": 25, "frame_idx": 50})
                archive.writestr(
                    f"map-keyframes/{video_id}.csv",
                    buffer.getvalue().encode("utf-8-sig"),
                )
        return path

    def _write_keyframe_zip(self) -> Path:
        path = self.root / "Keyframes_L01.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for video_id in ("L01_V001", "L01_V002"):
                for ordinal in (1, 2):
                    image = Image.new("RGB", (320, 180), (ordinal * 80, 40, 90))
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    archive.writestr(
                        f"keyframes/{video_id}/{ordinal:03d}.jpg", buffer.getvalue()
                    )
        return path

    def test_retrieval_mapping_selective_extract_and_sheet(self) -> None:
        feature_zip = self._write_feature_zip()
        mapping_zip = self._write_mapping_zip()
        self._write_keyframe_zip()
        retriever = FeatureZipRetriever(
            feature_zip=feature_zip,
            top_per_video=2,
            shortlist_size=3,
            max_candidates_per_video=2,
        )
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        candidates = retriever.retrieve(embeddings, 1.0, 0.0)
        self.assertEqual(candidates[0].candidate_id, "L01_V001__K001")

        mapping = KeyframeMapping(mapping_zip)
        assets = SelectiveAssetStore(
            data_root=self.root,
            keyframe_zip_glob="Keyframes_*.zip",
            video_zip_glob="Videos_*.zip",
            cache_dir=self.root / "cache",
            mapping=mapping,
        )
        for candidate in candidates:
            mapping.enrich(candidate)
            assets.extract_keyframe(candidate)
            self.assertTrue(candidate.image_path and candidate.image_path.is_file())

        nearest = mapping.nearest_time("L01_V001", 1.7)
        self.assertEqual(nearest.ordinal, 2)
        self.assertEqual(nearest.frame_idx, 50)

        builder = ContactSheetBuilder(
            ContactSheetConfig(columns=2, cards_per_sheet=3, card_width=320, card_height=230, image_height=170)
        )
        cards = [builder.candidate_card(candidate) for candidate in candidates]
        sheets = builder.build(cards, self.root / "sheets", "test")
        self.assertEqual(len(sheets), 1)
        self.assertTrue(sheets[0].is_file())

    def test_full_orchestration_with_fake_openai(self) -> None:
        feature_zip = self._write_feature_zip()
        mapping_zip = self._write_mapping_zip()
        self._write_keyframe_zip()

        class FakeEncoder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

        class FakeAPI:
            def plan_query(self, query: str, max_support_queries: int) -> QueryPlan:
                return QueryPlan(query, "complete visual query", must_have=["subject"])

            def rank_contact_sheets(self, query, plan, candidates, sheet_paths, limit):
                return {
                    "sufficient_match": True,
                    "selected": [
                        {
                            "candidate_id": candidates[0].candidate_id,
                            "match_score": 99,
                            "satisfied": ["subject"],
                            "missing": [],
                            "reason": "visible",
                        }
                    ],
                    "refined_retrieval_query_en": "",
                    "missing_focus": "",
                }

            def select_final(self, query, plan, candidates, sheet_paths):
                return {
                    "verified": True,
                    "selected_candidate_id": candidates[0].candidate_id,
                    "confidence": 0.9,
                    "satisfied": ["subject"],
                    "missing": [],
                    "reason": "best neighbor",
                }

        cfg = AppConfig(
            data=DataConfig(
                root=self.root,
                feature_zip=feature_zip,
                mapping_zip=mapping_zip,
                keyframe_zip_glob="Keyframes_*.zip",
                video_zip_glob="Videos_*.zip",
                cache_dir=self.root / "cache2",
                output_dir=self.root / "outputs",
            ),
            retrieval=RetrievalConfig(
                top_per_video=2,
                shortlist_size=3,
                max_candidates_per_video=2,
                max_rounds=1,
            ),
            contact_sheet=ContactSheetConfig(
                columns=2,
                cards_per_sheet=4,
                card_width=320,
                card_height=230,
                image_height=170,
            ),
            refinement=RefinementConfig(
                finalists=1,
                neighbor_radius=1,
                use_raw_video_for_motion=False,
            ),
        )
        cfg.validate()
        cfg.data.cache_dir.mkdir()
        cfg.data.output_dir.mkdir()
        pipeline = KeyframeSearchPipeline(
            cfg, api_client=FakeAPI(), encoder=FakeEncoder()
        )
        result = pipeline.run("test query")
        self.assertTrue(result["verified"])
        self.assertEqual(result["kis_mode"], "static")
        self.assertEqual(result["video_id"], "L01_V001")
        self.assertIn(result["frame_idx"], (0, 50))
        self.assertTrue(Path(result["keyframe_path"]).is_file())

    def test_video_level_fallback_after_empty_primary_ranking(self) -> None:
        feature_zip = self._write_feature_zip()
        mapping_zip = self._write_mapping_zip()
        self._write_keyframe_zip()

        class FakeEncoder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

        class FakeAPI:
            def __init__(self) -> None:
                self.rank_calls = 0

            def plan_query(self, query: str, max_support_queries: int) -> QueryPlan:
                return QueryPlan(
                    query,
                    "strict visual query",
                    must_have=["subject"],
                    fallback_visual_query_en="broader visual query",
                )

            def rank_contact_sheets(self, query, plan, candidates, sheet_paths, limit):
                self.rank_calls += 1
                if self.rank_calls == 1:
                    return {
                        "sufficient_match": False,
                        "selected": [],
                        "refined_retrieval_query_en": "",
                        "missing_focus": "not in primary shortlist",
                    }
                return {
                    "sufficient_match": True,
                    "selected": [
                        {
                            "candidate_id": candidates[0].candidate_id,
                            "match_score": 95,
                            "satisfied": ["subject"],
                            "missing": [],
                            "reason": "deep scan match",
                        }
                    ],
                    "refined_retrieval_query_en": "",
                    "missing_focus": "",
                }

            def select_videos(self, query, plan, representatives, sheet_paths, limit):
                return {
                    "sufficient_match": True,
                    "selected": [
                        {
                            "video_id": "L01_V001",
                            "match_score": 90,
                            "reason": "scene is present",
                        }
                    ],
                    "reason": "scan this video",
                }

            def select_local_anchors(self, query, plan, candidates, sheet_paths, limit):
                return {
                    "sufficient_match": True,
                    "selected": [
                        {
                            "candidate_id": candidates[0].candidate_id,
                            "match_score": 90,
                            "satisfied": ["subject"],
                            "missing": [],
                            "reason": "inspect this window",
                        }
                    ],
                    "refined_retrieval_query_en": "",
                    "missing_focus": "",
                }

            def select_final(self, query, plan, candidates, sheet_paths):
                return {
                    "verified": True,
                    "selected_candidate_id": candidates[0].candidate_id,
                    "confidence": 0.9,
                    "satisfied": ["subject"],
                    "missing": [],
                    "reason": "final deep-scan neighbor",
                }

        cfg = AppConfig(
            data=DataConfig(
                root=self.root,
                feature_zip=feature_zip,
                mapping_zip=mapping_zip,
                keyframe_zip_glob="Keyframes_*.zip",
                video_zip_glob="Videos_*.zip",
                cache_dir=self.root / "cache3",
                output_dir=self.root / "outputs3",
            ),
            retrieval=RetrievalConfig(
                top_per_video=2,
                shortlist_size=3,
                max_candidates_per_video=2,
                max_rounds=1,
            ),
            contact_sheet=ContactSheetConfig(
                columns=2,
                cards_per_sheet=4,
                card_width=320,
                card_height=230,
                image_height=170,
            ),
            refinement=RefinementConfig(
                finalists=1,
                neighbor_radius=1,
                use_raw_video_for_motion=False,
            ),
            fallback=FallbackConfig(
                relaxed_shortlist_size=2,
                candidate_videos=2,
                representative_frames_per_video=2,
                selected_videos_for_deep_scan=1,
                local_scan_anchors_per_video=1,
                local_scan_radius=1,
                deep_scan_batch_size=2,
                deep_scan_keep_per_video=1,
            ),
        )
        cfg.validate()
        cfg.data.cache_dir.mkdir()
        cfg.data.output_dir.mkdir()
        result = KeyframeSearchPipeline(
            cfg, api_client=FakeAPI(), encoder=FakeEncoder()
        ).run("test fallback query")

        self.assertTrue(result["verified"])
        self.assertEqual(result["strategy"], "video_level_fallback")
        self.assertEqual(result["video_id"], "L01_V001")
        self.assertEqual(result["fallback"]["selected_videos"], ["L01_V001"])
        self.assertTrue(
            (Path(result["run_dir"]) / "fallback_video_scan" / "local_scan" / "L01_V001").is_dir()
        )

    def test_rescue_zoom_verifies_strong_primary_video_without_fallback(self) -> None:
        feature_zip = self._write_feature_zip()
        mapping_zip = self._write_mapping_zip()
        self._write_keyframe_zip()

        class FakeEncoder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

        class FakeAPI:
            def plan_query(self, query: str, max_support_queries: int) -> QueryPlan:
                return QueryPlan(
                    query,
                    "complete visual query",
                    must_have=["subject", "small visible text"],
                    needs_text_reading=True,
                )

            def rank_contact_sheets(self, query, plan, candidates, sheet_paths, limit):
                return {
                    "sufficient_match": False,
                    "selected": [
                        {
                            "candidate_id": candidates[0].candidate_id,
                            "match_score": 88,
                            "satisfied": ["subject"],
                            "missing": ["small visible text"],
                            "reason": "needs a larger view",
                        }
                    ],
                    "refined_retrieval_query_en": "",
                    "missing_focus": "small text",
                }

            def select_final(self, query, plan, candidates, sheet_paths):
                return {
                    "verified": False,
                    "selected_candidate_id": None,
                    "confidence": 0.8,
                    "satisfied": ["subject"],
                    "missing": ["small visible text"],
                    "reason": "normal contact sheet is too small",
                }

            def select_rescue_zoom(self, query, plan, candidates, sheet_paths):
                return {
                    "verified": True,
                    "selected_candidate_id": candidates[0].candidate_id,
                    "confidence": 0.91,
                    "satisfied": ["subject", "small visible text"],
                    "missing": [],
                    "reason": "visible in the high-detail evidence",
                }

        cfg = AppConfig(
            data=DataConfig(
                root=self.root,
                feature_zip=feature_zip,
                mapping_zip=mapping_zip,
                keyframe_zip_glob="Keyframes_*.zip",
                video_zip_glob="Videos_*.zip",
                cache_dir=self.root / "cache4",
                output_dir=self.root / "outputs4",
            ),
            retrieval=RetrievalConfig(
                top_per_video=2,
                shortlist_size=3,
                max_candidates_per_video=2,
                max_rounds=1,
            ),
            contact_sheet=ContactSheetConfig(
                columns=2,
                cards_per_sheet=4,
                card_width=320,
                card_height=230,
                image_height=170,
            ),
            refinement=RefinementConfig(
                finalists=1,
                neighbor_radius=1,
                use_raw_video_for_motion=False,
            ),
            rescue_zoom=RescueZoomConfig(
                min_primary_score=75,
                preserve_primary_videos=1,
                max_candidates=4,
                neighbor_radius=1,
                columns=2,
                cards_per_sheet=2,
                card_width=420,
                card_height=300,
                image_height=220,
                text_detail_regions=["bottom"],
            ),
            fallback=FallbackConfig(enabled=False),
        )
        cfg.validate()
        cfg.data.cache_dir.mkdir()
        cfg.data.output_dir.mkdir()
        result = KeyframeSearchPipeline(
            cfg, api_client=FakeAPI(), encoder=FakeEncoder()
        ).run("test rescue zoom")

        rescue_dir = Path(result["run_dir"]) / "rescue_zoom"
        self.assertTrue(result["verified"])
        self.assertEqual(result["strategy"], "rescue_zoom")
        self.assertTrue(list(rescue_dir.glob("zoom_full_*.jpg")))
        self.assertTrue(list(rescue_dir.glob("zoom_detail_*.jpg")))

    def test_fallback_keeps_strong_primary_video_for_local_scan(self) -> None:
        feature_zip = self._write_feature_zip()
        mapping_zip = self._write_mapping_zip()
        self._write_keyframe_zip()

        class FakeEncoder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[0.0, 1.0] for _ in texts], dtype=np.float32)

        class FakeAPI:
            def __init__(self) -> None:
                self.rank_calls = 0
                self.final_calls = 0

            def plan_query(self, query: str, max_support_queries: int) -> QueryPlan:
                return QueryPlan(
                    query,
                    "primary visual query",
                    must_have=["subject"],
                    fallback_visual_query_en="broad visual query",
                )

            def rank_contact_sheets(self, query, plan, candidates, sheet_paths, limit):
                self.rank_calls += 1
                return {
                    "sufficient_match": False,
                    "selected": [
                        {
                            "candidate_id": candidates[0].candidate_id,
                            "match_score": 90 if self.rank_calls == 1 else 80,
                            "satisfied": ["subject"],
                            "missing": [],
                            "reason": "primary evidence" if self.rank_calls == 1 else "local match",
                        }
                    ],
                    "refined_retrieval_query_en": "",
                    "missing_focus": "",
                }

            def select_final(self, query, plan, candidates, sheet_paths):
                self.final_calls += 1
                if self.final_calls == 1:
                    return {
                        "verified": False,
                        "selected_candidate_id": None,
                        "confidence": 0.8,
                        "satisfied": ["subject"],
                        "missing": ["detail"],
                        "reason": "needs rescue/fallback",
                    }
                selected = next(
                    item for item in candidates if item.video_id == "L01_V002"
                )
                return {
                    "verified": True,
                    "selected_candidate_id": selected.candidate_id,
                    "confidence": 0.9,
                    "satisfied": ["subject"],
                    "missing": [],
                    "reason": "preserved primary video wins",
                }

            def select_rescue_zoom(self, query, plan, candidates, sheet_paths):
                return {
                    "verified": False,
                    "selected_candidate_id": None,
                    "confidence": 0.5,
                    "satisfied": ["subject"],
                    "missing": ["detail"],
                    "reason": "continue to local fallback",
                }

            def select_videos(self, query, plan, representatives, sheet_paths, limit):
                return {
                    "sufficient_match": True,
                    "selected": [
                        {
                            "video_id": "L01_V001",
                            "match_score": 90,
                            "reason": "fallback would otherwise choose this video",
                        }
                    ],
                    "reason": "choose V001",
                }

            def select_local_anchors(self, query, plan, candidates, sheet_paths, limit):
                return {
                    "sufficient_match": True,
                    "selected": [
                        {
                            "candidate_id": candidates[0].candidate_id,
                            "match_score": 80,
                            "satisfied": ["subject"],
                            "missing": [],
                            "reason": "local anchor",
                        }
                    ],
                    "refined_retrieval_query_en": "",
                    "missing_focus": "",
                }

        cfg = AppConfig(
            data=DataConfig(
                root=self.root,
                feature_zip=feature_zip,
                mapping_zip=mapping_zip,
                keyframe_zip_glob="Keyframes_*.zip",
                video_zip_glob="Videos_*.zip",
                cache_dir=self.root / "cache5",
                output_dir=self.root / "outputs5",
            ),
            retrieval=RetrievalConfig(
                top_per_video=2,
                shortlist_size=3,
                max_candidates_per_video=2,
                max_rounds=1,
            ),
            contact_sheet=ContactSheetConfig(
                columns=2,
                cards_per_sheet=4,
                card_width=320,
                card_height=230,
                image_height=170,
            ),
            refinement=RefinementConfig(
                finalists=1,
                neighbor_radius=1,
                use_raw_video_for_motion=False,
            ),
            rescue_zoom=RescueZoomConfig(
                preserve_primary_videos=1,
                max_candidates=4,
                neighbor_radius=1,
                cards_per_sheet=2,
                card_width=420,
                card_height=300,
                image_height=220,
            ),
            fallback=FallbackConfig(
                relaxed_shortlist_size=2,
                candidate_videos=2,
                representative_frames_per_video=2,
                selected_videos_for_deep_scan=2,
                preserve_primary_videos=1,
                local_scan_anchors_per_video=1,
                local_scan_radius=1,
                deep_scan_batch_size=2,
                deep_scan_keep_per_video=1,
            ),
        )
        cfg.validate()
        cfg.data.cache_dir.mkdir()
        cfg.data.output_dir.mkdir()
        result = KeyframeSearchPipeline(
            cfg, api_client=FakeAPI(), encoder=FakeEncoder()
        ).run("test preserve primary")

        self.assertTrue(result["verified"])
        self.assertEqual(result["strategy"], "video_level_fallback")
        self.assertEqual(result["video_id"], "L01_V002")
        self.assertEqual(result["fallback"]["selected_videos"][0], "L01_V002")

    def test_typed_hard_gate_blocks_exact_text_but_not_context(self) -> None:
        pipeline = object.__new__(KeyframeSearchPipeline)
        pipeline.config = SimpleNamespace(verification=VerificationConfig())
        plan = QueryPlan(
            "query",
            "query",
            criteria=[
                Criterion(
                    "c1", "exact_text", "Countdown reads 13", "13", True, 0.99
                ),
                Criterion(
                    "c2", "context", "The route is Hồ Tùng Mậu", "", True, 0.99
                ),
            ],
        )
        decision = {
            "verified": True,
            "criterion_results": [
                {"criterion_id": "c1", "satisfied": False, "confidence": 0.9},
                {"criterion_id": "c2", "satisfied": False, "confidence": 0.9},
            ],
            "ranked_candidates": [],
        }
        pipeline._apply_hard_gate(plan, decision)
        self.assertFalse(decision["verified"])
        self.assertEqual(decision["hard_gate_failures"][0]["criterion_id"], "c1")

        decision["verified"] = True
        decision["criterion_results"][0]["satisfied"] = True
        pipeline._apply_hard_gate(plan, decision)
        self.assertTrue(decision["verified"])
        self.assertTrue(decision["hard_gate_passed"])

    def test_multi_event_normalization_requires_chronological_frames(self) -> None:
        pipeline = object.__new__(KeyframeSearchPipeline)
        pipeline.config = SimpleNamespace(verification=VerificationConfig())
        criterion1 = Criterion("E1_c1", "visual", "durian", required=True)
        criterion2 = Criterion("E2_c1", "visual", "mangosteen", required=True)
        plan = QueryPlan(
            "sequence query",
            "orchard sequence",
            events=[
                EventPlan("E1", "sầu riêng", "durian in an orchard", [criterion1]),
                EventPlan("E2", "măng cụt", "mangosteen in an orchard", [criterion2]),
            ],
            sequence_required=True,
        )
        candidates = [
            Candidate(
                "L01_V001__K001", "L01_V001", 1, 0.9, 0.9,
                frame_idx=0, pts_time=1.0, fps=25.0, image_path=Path("1.jpg")
            ),
            Candidate(
                "L01_V001__K002", "L01_V001", 2, 0.9, 0.9,
                frame_idx=50, pts_time=2.0, fps=25.0, image_path=Path("2.jpg")
            ),
        ]

        def event_row(event_id: str, candidate_id: str, criterion_id: str):
            evidence = {
                "criterion_id": criterion_id,
                "satisfied": True,
                "confidence": 0.95,
                "evidence": "visible",
            }
            ranked = {
                "candidate_id": candidate_id,
                "verified": True,
                "confidence": 0.95,
                "satisfied": ["visible"],
                "missing": [],
                "reason": "visible",
                "criterion_results": [evidence],
            }
            return {
                "event_id": event_id,
                "selected_candidate_id": candidate_id,
                "verified": True,
                "confidence": 0.95,
                "satisfied": ["visible"],
                "missing": [],
                "reason": "visible",
                "criterion_results": [evidence],
                "ranked_candidates": [ranked],
            }

        decision = {
            "verified": True,
            "confidence": 0.95,
            "reason": "ordered",
            "events": [
                event_row("E1", "L01_V001__K001", "E1_c1"),
                event_row("E2", "L01_V001__K002", "E2_c1"),
            ],
        }
        normalized = pipeline._normalize_multi_event_decision(
            "query", plan, "L01_V001", candidates, decision
        )
        self.assertTrue(normalized["verified"])
        self.assertTrue(normalized["chronological"])
        self.assertEqual(normalized["event_coverage"], 2)

        decision["events"] = [
            event_row("E1", "L01_V001__K002", "E1_c1"),
            event_row("E2", "L01_V001__K001", "E2_c1"),
        ]
        normalized = pipeline._normalize_multi_event_decision(
            "query", plan, "L01_V001", candidates, decision
        )
        self.assertFalse(normalized["verified"])
        self.assertFalse(normalized["chronological"])
        self.assertEqual(normalized["event_coverage"], 2)

    def test_explicit_task_type_normalization(self) -> None:
        self.assertEqual(KeyframeSearchPipeline._normalize_task_type("KIS"), "kis")
        self.assertEqual(
            KeyframeSearchPipeline._normalize_task_type("TRAKE"), "trake"
        )
        self.assertEqual(KeyframeSearchPipeline._normalize_task_type("QA"), "qa")
        self.assertEqual(KeyframeSearchPipeline._normalize_task_type(None), "auto")
        with self.assertRaises(ValueError):
            KeyframeSearchPipeline._normalize_task_type("unknown")

    def test_qa_mode_normalization_and_inference(self) -> None:
        self.assertEqual(
            KeyframeSearchPipeline._normalize_qa_mode("LONG_RANGE_TEMPORAL"),
            "long_range_temporal",
        )
        self.assertEqual(
            KeyframeSearchPipeline._normalize_qa_mode("not_applicable"), "auto"
        )
        with self.assertRaises(ValueError):
            KeyframeSearchPipeline._normalize_qa_mode("unknown")

        single = QueryPlan("number on car?", "number on car", task_type="qa")
        self.assertEqual(KeyframeSearchPipeline._infer_qa_mode(single), "single_frame")

        short = QueryPlan(
            "four fish across a cooking segment",
            "chef prepares four fish",
            task_type="qa",
            needs_motion=True,
        )
        self.assertEqual(KeyframeSearchPipeline._infer_qa_mode(short), "short_window")

        long = QueryPlan(
            "First a donation, then an elderly man talks to foreigners. Which road?",
            "charity lodging story",
            task_type="qa",
            events=[
                EventPlan("E1", "trao quà", "people donate supplies"),
                EventPlan("E2", "nói chuyện", "old man talks to foreigners"),
            ],
        )
        self.assertEqual(
            KeyframeSearchPipeline._infer_qa_mode(long), "long_range_temporal"
        )

    def test_qa_normalization_accepts_video_scope_count(self) -> None:
        pipeline = object.__new__(KeyframeSearchPipeline)
        pipeline.config = SimpleNamespace(
            verification=VerificationConfig(), qa=QAConfig()
        )
        plan = QueryPlan(
            "four fish are prepared; what fish is it?",
            "chef stuffs four fish",
            criteria=[
                Criterion(
                    "c_count",
                    "count",
                    "A total of four fish are prepared in the video.",
                    "4",
                    True,
                    0.99,
                    evidence_scope="video",
                )
            ],
            task_type="qa",
            qa_question="What fish is it?",
            target_object="the prepared fish",
            answer_expected_visible=False,
        )
        candidate = Candidate(
            "L01_V001__K001",
            "L01_V001",
            1,
            0.9,
            0.9,
            frame_idx=0,
            pts_time=0.0,
            fps=25.0,
            image_path=Path("fish.jpg"),
        )
        row = {
            "video_id": "L01_V001",
            "answer": None,
            "answer_source": "not_visible",
            "verified": True,
            "confidence": 0.9,
            "target_object_visible": True,
            "count_status": "across_frames",
            "evidence_candidate_ids": [candidate.candidate_id],
            "criterion_results": [
                {
                    "criterion_id": "c_count",
                    "satisfied": True,
                    "confidence": 0.9,
                    "evidence": "The count is accumulated across the sequence.",
                }
            ],
            "missing": [],
            "reason": "Video-level count is supported.",
        }
        result = pipeline._normalize_qa_result(plan, row, {candidate.candidate_id: candidate})
        self.assertTrue(result["verified"])
        self.assertEqual(result["count_status"], "across_frames")
        self.assertEqual(result["evidence_frames"][0]["frame_idx"], 0)

    def test_explicit_qa_route_returns_five_review_slots(self) -> None:
        feature_zip = self._write_feature_zip()
        mapping_zip = self._write_mapping_zip()
        self._write_keyframe_zip()

        class FakeEncoder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

        class FakeAPI:
            def __init__(self) -> None:
                self.screen_calls = 0

            def plan_query(self, query: str, max_support_queries: int) -> QueryPlan:
                return QueryPlan(
                    query,
                    "four prepared fish",
                    task_type="qa",
                    qa_question="What fish is it?",
                    target_object="prepared fish",
                )

            def screen_qa_videos(self, query, plan, candidates, sheet_paths, limit):
                self.screen_calls += 1
                video_ids = []
                for candidate in candidates:
                    if candidate.video_id not in video_ids:
                        video_ids.append(candidate.video_id)
                return {
                    "sufficient_match": True,
                    "selected": [
                        {
                            "video_id": video_id,
                            "match_score": 80,
                            "reason": "screened",
                        }
                        for video_id in video_ids[:limit]
                    ],
                    "reason": "screening fixture",
                }

            def select_qa_evidence(self, query, plan, candidates, sheet_paths, result_count):
                rows = []
                for candidate in candidates[:2]:
                    rows.append(
                        {
                            "video_id": candidate.video_id,
                            "answer": None,
                            "answer_source": "not_visible",
                            "verified": True,
                            "confidence": 0.9,
                            "target_object_visible": True,
                            "count_status": "not_verified",
                            "evidence_candidate_ids": [candidate.candidate_id],
                            "criterion_results": [],
                            "missing": [],
                            "reason": "review evidence",
                        }
                    )
                return {"results": rows, "reason": "two videos in fixture"}

        cfg = AppConfig(
            data=DataConfig(
                root=self.root,
                feature_zip=feature_zip,
                mapping_zip=mapping_zip,
                keyframe_zip_glob="Keyframes_*.zip",
                video_zip_glob="Videos_*.zip",
                cache_dir=self.root / "qa_cache",
                output_dir=self.root / "qa_outputs",
            ),
            retrieval=RetrievalConfig(top_per_video=2, shortlist_size=3),
            contact_sheet=ContactSheetConfig(
                columns=2, cards_per_sheet=4, card_width=320, card_height=230, image_height=170
            ),
        )
        cfg.validate()
        cfg.data.cache_dir.mkdir()
        cfg.data.output_dir.mkdir()
        api = FakeAPI()
        result = KeyframeSearchPipeline(
            cfg, api_client=api, encoder=FakeEncoder()
        ).run("four fish; what fish is it?", task_type="qa")
        self.assertEqual(api.screen_calls, 1)
        self.assertEqual(result["task_type"], "qa")
        self.assertEqual(result["qa_mode"], "single_frame")
        self.assertEqual(result["qa_mode_source"], "local_heuristic")
        self.assertEqual(len(result["results"]), 5)
        self.assertTrue(result["results"][0]["verified"])
        self.assertEqual(result["results"][0]["evidence_frames"][0]["frame_idx"], 0)

        forced = KeyframeSearchPipeline(
            cfg, api_client=api, encoder=FakeEncoder()
        ).run(
            "first a donation, then an elderly man talks to foreigners; which road?",
            task_type="qa",
            qa_mode="long_range_temporal",
        )
        self.assertEqual(forced["qa_mode"], "long_range_temporal")
        self.assertEqual(forced["qa_mode_source"], "run_argument")

    def test_qa_coverage_prefers_video_matching_multiple_criteria(self) -> None:
        feature_zip = self.root / "qa_coverage_features.zip"
        arrays = {
            "clip-features-32/L01_V001.npy": np.array(
                [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32
            ),
            "clip-features-32/L01_V002.npy": np.array(
                [[0.6, 0.8, 0.0], [0.6, 0.0, 0.8]], dtype=np.float32
            ),
        }
        with zipfile.ZipFile(feature_zip, "w") as archive:
            for member, array in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, array)
                archive.writestr(member, buffer.getvalue())
        retriever = FeatureZipRetriever(feature_zip, 2, 10, 4)
        _, ranking = retriever.retrieve_video_coverage(
            np.eye(3, dtype=np.float32),
            ["main", "ingredient_a", "ingredient_b"],
            top_frames_per_query=1,
            coverage_rank_cutoff=1,
            video_limit=2,
            coverage_weight=0.5,
            evidence_score_weight=0.3,
            main_query_weight=0.2,
        )
        self.assertEqual(ranking[0]["video_id"], "L01_V002")
        self.assertEqual(ranking[0]["coverage_count"], 2)
        self.assertEqual(ranking[1]["coverage_count"], 0)

        _, rescued = retriever.retrieve_video_coverage(
            np.eye(3, dtype=np.float32),
            ["main", "ingredient_a", "ingredient_b"],
            top_frames_per_query=1,
            coverage_rank_cutoff=1,
            video_limit=1,
            coverage_weight=0.5,
            evidence_score_weight=0.3,
            main_query_weight=0.2,
            preserve_main_query_videos=1,
        )
        self.assertIn("L01_V001", {row["video_id"] for row in rescued})

    def test_qa_fallback_opens_second_video_batch(self) -> None:
        pipeline = object.__new__(KeyframeSearchPipeline)
        pipeline.config = SimpleNamespace(
            qa=QAConfig(), verification=VerificationConfig()
        )
        candidates = [
            Candidate(
                f"L01_V{index:03d}__K001",
                f"L01_V{index:03d}",
                1,
                1.0 - index / 100.0,
                1.0 - index / 100.0,
                frame_idx=index * 25,
                pts_time=float(index),
                fps=25.0,
                image_path=self.root / f"{index}.jpg",
            )
            for index in range(1, 11)
        ]
        ranking = [
            {"video_id": candidate.video_id, "joint_score": candidate.retrieval_score}
            for candidate in candidates
        ]
        pipeline._prepare_qa_candidates = lambda plan, run_dir, **kwargs: (
            candidates,
            [candidate.video_id for candidate in candidates],
            ranking,
        )
        pipeline._candidate_sheets = lambda candidates, directory, prefix: []

        class FakeQAAPI:
            def __init__(self) -> None:
                self.calls = 0

            def select_qa_evidence(self, query, plan, batch, sheets, result_count):
                self.calls += 1
                rows = []
                for candidate in batch:
                    is_winner = self.calls == 2 and candidate.video_id == "L01_V006"
                    rows.append(
                        {
                            "video_id": candidate.video_id,
                            "answer": "fish" if is_winner else None,
                            "answer_source": "visual_inference" if is_winner else "not_visible",
                            "verified": is_winner,
                            "confidence": 0.9 if is_winner else 0.4,
                            "target_object_visible": True,
                            "count_status": "not_applicable",
                            "evidence_candidate_ids": [candidate.candidate_id],
                            "criterion_results": [],
                            "missing": [] if is_winner else ["full match"],
                            "reason": "winner" if is_winner else "partial",
                        }
                    )
                return {"results": rows, "reason": f"round {self.calls}"}

        pipeline.api = FakeQAAPI()
        plan = QueryPlan(
            "question", "complete visual query", task_type="qa", qa_question="what?"
        )
        result = pipeline._run_qa("question", plan, self.root / "qa_rounds")
        self.assertEqual(len(result["qa_attempts"]), 2)
        self.assertEqual(result["qa_attempts"][1]["video_ids"][0], "L01_V006")
        self.assertTrue(result["verified"])
        self.assertEqual(result["results"][0]["video_id"], "L01_V006")

    def test_temporal_kis_detection_is_scope_aware(self) -> None:
        static_plan = QueryPlan(
            "static",
            "static",
            criteria=[Criterion("c1", "visual", "five people", evidence_scope="frame")],
        )
        self.assertFalse(KeyframeSearchPipeline._requires_temporal_kis(static_plan))

        window_plan = QueryPlan(
            "window",
            "window",
            criteria=[
                Criterion(
                    "c1", "visual", "a person hides an object", evidence_scope="window"
                )
            ],
        )
        self.assertTrue(KeyframeSearchPipeline._requires_temporal_kis(window_plan))

        motion_plan = QueryPlan("motion", "motion", needs_motion=True)
        self.assertTrue(KeyframeSearchPipeline._requires_temporal_kis(motion_plan))

    def test_temporal_kis_fallback_verifies_video_then_representative_frame(self) -> None:
        pipeline = object.__new__(KeyframeSearchPipeline)
        pipeline.config = SimpleNamespace(
            temporal_kis=TemporalKISConfig(screening_enabled=False),
            verification=VerificationConfig(),
            openai=OpenAIConfig(final_candidate_count=5),
        )
        candidates = [
            Candidate(
                f"L01_V{video:03d}__K{ordinal:03d}",
                f"L01_V{video:03d}",
                ordinal,
                1.0 - video / 100.0,
                1.0 - video / 100.0,
                frame_idx=video * 250 + ordinal * 25,
                pts_time=float(video * 10 + ordinal),
                fps=25.0,
                image_path=self.root / f"temporal_{video}_{ordinal}.jpg",
            )
            for video in range(1, 11)
            for ordinal in (1, 2)
        ]
        ranking = [
            {"video_id": f"L01_V{video:03d}", "joint_score": 1.0 - video / 100.0}
            for video in range(1, 11)
        ]
        pipeline._prepare_qa_candidates = lambda plan, run_dir, **kwargs: (
            candidates,
            [f"L01_V{video:03d}" for video in range(1, 11)],
            ranking,
        )
        pipeline._candidate_sheets = lambda candidates, directory, prefix: []

        class FakeTemporalAPI:
            def __init__(self) -> None:
                self.calls = 0

            def select_temporal_kis(self, query, plan, batch, sheets, result_count):
                self.calls += 1
                selected = batch[0]
                verified = self.calls == 2
                criterion_results = [
                    {
                        "criterion_id": "c_action",
                        "satisfied": verified,
                        "confidence": 0.95 if verified else 0.3,
                        "evidence": "action visible" if verified else "action missing",
                        "evidence_candidate_ids": [selected.candidate_id] if verified else [],
                    },
                    {
                        "criterion_id": "c_order",
                        "satisfied": verified,
                        "confidence": 0.95 if verified else 0.3,
                        "evidence": "ordered" if verified else "order missing",
                        "evidence_candidate_ids": (
                            [batch[0].candidate_id, batch[1].candidate_id]
                            if verified
                            else []
                        ),
                    },
                ]
                return {
                    "verified": verified,
                    "video_id": selected.video_id,
                    "selected_candidate_id": selected.candidate_id,
                    "confidence": 0.9 if verified else 0.4,
                    "satisfied": ["sequence"] if verified else [],
                    "missing": [] if verified else ["sequence"],
                    "reason": "verified sequence" if verified else "partial sequence",
                    "criterion_results": criterion_results,
                    "ranked_candidates": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "confidence": 0.8,
                            "reason": "review frame",
                        }
                        for candidate in batch
                    ],
                }

        pipeline.api = FakeTemporalAPI()
        plan = QueryPlan(
            "story query",
            "five people, a yellow animal, and a hidden pumpkin",
            criteria=[
                Criterion(
                    "c_action",
                    "visual",
                    "A person hides a pumpkin-like object.",
                    evidence_scope="window",
                ),
                Criterion(
                    "c_order",
                    "temporal",
                    "The hiding happens before the man wakes the animal.",
                    evidence_scope="video",
                ),
            ],
            needs_motion=True,
            task_type="kis",
        )
        result = pipeline._run_temporal_kis(
            "story query", plan, self.root / "temporal_kis_run"
        )
        self.assertEqual(len(result["temporal_kis_attempts"]), 2)
        self.assertTrue(result["verified"])
        self.assertTrue(result["video_verified"])
        self.assertTrue(result["frame_verified"])
        self.assertEqual(result["kis_mode"], "temporal")
        self.assertEqual(result["video_id"], "L01_V006")


if __name__ == "__main__":
    unittest.main()
