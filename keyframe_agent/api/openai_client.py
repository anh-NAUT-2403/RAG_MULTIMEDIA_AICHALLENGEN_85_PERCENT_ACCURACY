from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..config import OpenAIConfig
from ..models import Candidate, Criterion, EventPlan, QueryPlan, SampleFrame
from .prompts import (
    contact_rank_prompt,
    final_select_prompt,
    local_anchor_prompt,
    multi_event_select_prompt,
    qa_evidence_prompt,
    qa_video_screen_prompt,
    query_plan_prompt,
    rescue_zoom_prompt,
    sample_select_prompt,
    temporal_kis_select_prompt,
    video_rank_prompt,
)
from .schemas import (
    CONTACT_RANK_SCHEMA,
    FINAL_SELECT_SCHEMA,
    MULTI_EVENT_SELECT_SCHEMA,
    QA_EVIDENCE_SCHEMA,
    QUERY_PLAN_SCHEMA,
    SAMPLE_SELECT_SCHEMA,
    TEMPORAL_KIS_SCHEMA,
    VIDEO_SELECT_SCHEMA,
)


class OpenAIKeyframeClient:
    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {config.api_key_env}"
            )
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _image_data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    def _request_json(
        self,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        image_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for path in image_paths or []:
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._image_data_url(path),
                    "detail": self.config.image_detail,
                }
            )

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
        }
        if self.config.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort}

        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                response = self.client.responses.create(**kwargs)
                return json.loads(response.output_text)
            except Exception as error:  # SDK/network errors vary by version.
                last_error = error
                if attempt + 1 >= self.config.retries:
                    break
                time.sleep(self.config.retry_base_seconds * (2**attempt))
        raise RuntimeError(f"OpenAI request failed after retries: {last_error}")

    def plan_query(self, query: str, max_support_queries: int) -> QueryPlan:
        result = self._request_json(
            query_plan_prompt(query), "keyframe_query_plan", QUERY_PLAN_SCHEMA
        )
        criteria = [Criterion(**item) for item in result["criteria"]]
        events = [
            EventPlan(
                event_id=item["event_id"],
                description_vi=item["description_vi"],
                visual_query_en=item["visual_query_en"],
                criteria=[Criterion(**criterion) for criterion in item["criteria"]],
            )
            for item in result["events"]
        ]
        must_have = [
            item.description
            for item in criteria
            if item.required and item.criterion_type != "context"
        ]
        return QueryPlan(
            original_query=query,
            main_visual_query_en=result["main_visual_query_en"],
            fallback_visual_query_en=result["fallback_visual_query_en"],
            support_queries_en=result["support_queries_en"][:max_support_queries],
            must_have=must_have,
            optional=result["optional"],
            needs_motion=bool(result["needs_motion"]),
            needs_text_reading=bool(result["needs_text_reading"]),
            search_notes=result["search_notes"],
            criteria=criteria,
            events=events,
            sequence_required=bool(result["sequence_required"]),
            task_type=result["task_type"],
            qa_question=result["qa_question"],
            target_object=result["target_object"],
            answer_expected_visible=bool(result["answer_expected_visible"]),
            qa_mode=result["qa_mode"],
        )

    def rank_contact_sheets(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        sheet_paths: list[Path],
        limit: int,
    ) -> dict[str, Any]:
        return self._request_json(
            contact_rank_prompt(query, plan, candidates, limit),
            "contact_sheet_ranking",
            CONTACT_RANK_SCHEMA,
            sheet_paths,
        )

    def select_sample(
        self,
        query: str,
        plan: QueryPlan,
        samples: list[SampleFrame],
        sheet_paths: list[Path],
    ) -> dict[str, Any]:
        return self._request_json(
            sample_select_prompt(query, plan, samples),
            "temporal_sample_selection",
            SAMPLE_SELECT_SCHEMA,
            sheet_paths,
        )

    def select_videos(
        self,
        query: str,
        plan: QueryPlan,
        representatives: list[Candidate],
        sheet_paths: list[Path],
        limit: int,
    ) -> dict[str, Any]:
        return self._request_json(
            video_rank_prompt(query, plan, representatives, limit),
            "fallback_video_selection",
            VIDEO_SELECT_SCHEMA,
            sheet_paths,
        )

    def screen_qa_videos(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        sheet_paths: list[Path],
        limit: int,
    ) -> dict[str, Any]:
        return self._request_json(
            qa_video_screen_prompt(query, plan, candidates, limit),
            "qa_video_screening",
            VIDEO_SELECT_SCHEMA,
            sheet_paths,
        )

    def select_local_anchors(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        sheet_paths: list[Path],
        limit: int,
    ) -> dict[str, Any]:
        return self._request_json(
            local_anchor_prompt(query, plan, candidates, limit),
            "fallback_local_anchor_selection",
            CONTACT_RANK_SCHEMA,
            sheet_paths,
        )

    def select_rescue_zoom(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        sheet_paths: list[Path],
    ) -> dict[str, Any]:
        return self._request_json(
            rescue_zoom_prompt(
                query, plan, candidates, self.config.final_candidate_count
            ),
            "rescue_zoom_final_selection",
            FINAL_SELECT_SCHEMA,
            sheet_paths,
        )

    def select_final(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        sheet_paths: list[Path],
    ) -> dict[str, Any]:
        return self._request_json(
            final_select_prompt(
                query, plan, candidates, self.config.final_candidate_count
            ),
            "final_keyframe_selection",
            FINAL_SELECT_SCHEMA,
            sheet_paths,
        )

    def select_multi_event_sequence(
        self,
        query: str,
        plan: QueryPlan,
        video_id: str,
        candidates: list[Candidate],
        sheet_paths: list[Path],
    ) -> dict[str, Any]:
        return self._request_json(
            multi_event_select_prompt(
                query,
                plan,
                video_id,
                candidates,
                self.config.final_candidate_count,
            ),
            "multi_event_trake_selection",
            MULTI_EVENT_SELECT_SCHEMA,
            sheet_paths,
        )

    def select_qa_evidence(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        sheet_paths: list[Path],
        result_count: int,
    ) -> dict[str, Any]:
        return self._request_json(
            qa_evidence_prompt(query, plan, candidates, result_count),
            "qa_video_evidence",
            QA_EVIDENCE_SCHEMA,
            sheet_paths,
        )

    def select_temporal_kis(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        sheet_paths: list[Path],
        result_count: int,
    ) -> dict[str, Any]:
        return self._request_json(
            temporal_kis_select_prompt(query, plan, candidates, result_count),
            "temporal_kis_selection",
            TEMPORAL_KIS_SCHEMA,
            sheet_paths,
        )
