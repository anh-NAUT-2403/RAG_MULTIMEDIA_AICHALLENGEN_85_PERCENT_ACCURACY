QUERY_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "main_visual_query_en": {"type": "string"},
        "fallback_visual_query_en": {"type": "string"},
        "support_queries_en": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        },
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "criterion_id": {"type": "string"},
                    "criterion_type": {
                        "type": "string",
                        "enum": [
                            "visual",
                            "exact_text",
                            "text_presence",
                            "count",
                            "temporal",
                            "context",
                        ],
                    },
                    "description": {"type": "string"},
                    "value": {"type": "string"},
                    "required": {"type": "boolean"},
                    "classification_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "reason": {"type": "string"},
                    "evidence_scope": {
                        "type": "string",
                        "enum": ["frame", "window", "video"],
                    },
                },
                "required": [
                    "criterion_id",
                    "criterion_type",
                    "description",
                    "value",
                    "required",
                    "classification_confidence",
                    "reason",
                    "evidence_scope",
                ],
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_id": {"type": "string"},
                    "description_vi": {"type": "string"},
                    "visual_query_en": {"type": "string"},
                    "criteria": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "criterion_id": {"type": "string"},
                                "criterion_type": {
                                    "type": "string",
                                    "enum": [
                                        "visual",
                                        "exact_text",
                                        "text_presence",
                                        "count",
                                        "temporal",
                                        "context",
                                    ],
                                },
                                "description": {"type": "string"},
                                "value": {"type": "string"},
                                "required": {"type": "boolean"},
                                "classification_confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "reason": {"type": "string"},
                                "evidence_scope": {
                                    "type": "string",
                                    "enum": ["frame", "window", "video"],
                                },
                            },
                            "required": [
                                "criterion_id",
                                "criterion_type",
                                "description",
                                "value",
                                "required",
                                "classification_confidence",
                                "reason",
                                "evidence_scope",
                            ],
                        },
                    },
                },
                "required": [
                    "event_id",
                    "description_vi",
                    "visual_query_en",
                    "criteria",
                ],
            },
        },
        "sequence_required": {"type": "boolean"},
        "task_type": {"type": "string", "enum": ["kis", "trake", "qa"]},
        "qa_question": {"type": "string"},
        "target_object": {"type": "string"},
        "answer_expected_visible": {"type": "boolean"},
        "qa_mode": {
            "type": "string",
            "enum": [
                "not_applicable",
                "single_frame",
                "short_window",
                "long_range_temporal",
            ],
        },
        "optional": {"type": "array", "items": {"type": "string"}},
        "needs_motion": {"type": "boolean"},
        "needs_text_reading": {"type": "boolean"},
        "search_notes": {"type": "string"},
    },
    "required": [
        "main_visual_query_en",
        "fallback_visual_query_en",
        "support_queries_en",
        "criteria",
        "events",
        "sequence_required",
        "task_type",
        "qa_question",
        "target_object",
        "answer_expected_visible",
        "qa_mode",
        "optional",
        "needs_motion",
        "needs_text_reading",
        "search_notes",
    ],
}

VIDEO_SELECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sufficient_match": {"type": "boolean"},
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "video_id": {"type": "string"},
                    "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "reason": {"type": "string"},
                },
                "required": ["video_id", "match_score", "reason"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["sufficient_match", "selected", "reason"],
}

CONTACT_RANK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sufficient_match": {"type": "boolean"},
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "satisfied": {"type": "array", "items": {"type": "string"}},
                    "missing": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "match_score",
                    "satisfied",
                    "missing",
                    "reason",
                ],
            },
        },
        "refined_retrieval_query_en": {"type": "string"},
        "missing_focus": {"type": "string"},
    },
    "required": [
        "sufficient_match",
        "selected",
        "refined_retrieval_query_en",
        "missing_focus",
    ],
}

SAMPLE_SELECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_sample_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["selected_sample_id", "confidence", "reason"],
}

FINAL_SELECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verified": {"type": "boolean"},
        "selected_candidate_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "satisfied": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "criterion_results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "criterion_id": {"type": "string"},
                    "satisfied": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                },
                "required": ["criterion_id", "satisfied", "confidence", "evidence"],
            },
        },
        "ranked_candidates": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "verified": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "satisfied": {"type": "array", "items": {"type": "string"}},
                    "missing": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "criterion_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "criterion_id": {"type": "string"},
                                "satisfied": {"type": "boolean"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "evidence": {"type": "string"},
                            },
                            "required": [
                                "criterion_id",
                                "satisfied",
                                "confidence",
                                "evidence",
                            ],
                        },
                    },
                },
                "required": [
                    "candidate_id",
                    "verified",
                    "confidence",
                    "satisfied",
                    "missing",
                    "reason",
                    "criterion_results",
                ],
            },
        },
    },
    "required": [
        "verified",
        "selected_candidate_id",
        "confidence",
        "satisfied",
        "missing",
        "reason",
        "criterion_results",
        "ranked_candidates",
    ],
}

MULTI_EVENT_SELECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verified": {"type": "boolean"},
        "video_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_id": {"type": "string"},
                    "selected_candidate_id": {"type": ["string", "null"]},
                    "verified": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "satisfied": {"type": "array", "items": {"type": "string"}},
                    "missing": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "criterion_results": FINAL_SELECT_SCHEMA["properties"][
                        "criterion_results"
                    ],
                    "ranked_candidates": FINAL_SELECT_SCHEMA["properties"][
                        "ranked_candidates"
                    ],
                },
                "required": [
                    "event_id",
                    "selected_candidate_id",
                    "verified",
                    "confidence",
                    "satisfied",
                    "missing",
                    "reason",
                    "criterion_results",
                    "ranked_candidates",
                ],
            },
        },
    },
    "required": ["verified", "video_id", "confidence", "reason", "events"],
}

QA_EVIDENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "video_id": {"type": "string"},
                    "answer": {"type": ["string", "null"]},
                    "answer_source": {
                        "type": "string",
                        "enum": ["visible_text", "visual_inference", "not_visible"],
                    },
                    "verified": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "target_object_visible": {"type": "boolean"},
                    "count_status": {
                        "type": "string",
                        "enum": [
                            "same_frame",
                            "across_frames",
                            "not_verified",
                            "not_applicable",
                        ],
                    },
                    "evidence_candidate_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": {"type": "string"},
                    },
                    "answer_evidence_candidate_ids": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                    "criterion_results": FINAL_SELECT_SCHEMA["properties"][
                        "criterion_results"
                    ],
                    "missing": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": [
                    "video_id",
                    "answer",
                    "answer_source",
                    "verified",
                    "confidence",
                    "target_object_visible",
                    "count_status",
                    "evidence_candidate_ids",
                    "answer_evidence_candidate_ids",
                    "criterion_results",
                    "missing",
                    "reason",
                ],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["results", "reason"],
}

TEMPORAL_KIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verified": {"type": "boolean"},
        "video_id": {"type": ["string", "null"]},
        "selected_candidate_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "satisfied": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "criterion_results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "criterion_id": {"type": "string"},
                    "satisfied": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "evidence": {"type": "string"},
                    "evidence_candidate_ids": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "criterion_id",
                    "satisfied",
                    "confidence",
                    "evidence",
                    "evidence_candidate_ids",
                ],
            },
        },
        "ranked_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["candidate_id", "confidence", "reason"],
            },
        },
    },
    "required": [
        "verified",
        "video_id",
        "selected_candidate_id",
        "confidence",
        "satisfied",
        "missing",
        "reason",
        "criterion_results",
        "ranked_candidates",
    ],
}
