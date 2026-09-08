from __future__ import annotations

import json

from ..models import Candidate, QueryPlan, SampleFrame


def query_plan_prompt(query: str) -> str:
    return f"""
You prepare retrieval text for an AIC video-keyframe search system.

Original Vietnamese query:
{query}

Rules:
1. Preserve the complete subject + attributes + objects + actions + composition.
2. Write one faithful English visual query as the authoritative CLIP query.
3. Write one fallback English visual query for finding a likely source video when
   exact retrieval fails. Keep the main scene, subjects and composition, but remove
   tiny text, proper names, exact numbers and other details that CLIP cannot reliably
   encode. This query can only seed a later video-level scan; it must not verify a
   final answer.
4. Write at most two complete support queries. Never split isolated fragments such
   as only a person, only a color, or only an action.
5. Decompose the query into criteria with unique stable IDs c1, c2, ... and classify
   each criterion as exactly one of:
   - visual: visible subject, attribute, object, action or spatial composition;
   - exact_text: exact text/number must be readable because its content is requested;
   - text_presence: a sign/writing and its visual form or approximate character count
     must be visible, but exact transcription is not requested;
   - count: an exact visible count is decisive;
   - temporal: order, transition or brief motion is decisive;
   - context: a place/person/event name helps identify the source but the query does
     not say that its written name, sign or label must appear in the frame.
   Mark required=true only when the query explicitly makes the criterion necessary.
   Give classification_confidence and a short reason. Do not classify a named road,
   city or event as exact_text merely because its name occurs in the query. It becomes
   exact_text only when the query asks for that text/sign/label to be visible or read.
   Also set evidence_scope:
   - frame: must be shown in one keyframe;
   - window: needs several nearby frames to verify an action/order;
   - video: is an aggregate condition across the video, such as a total count.
   Do not turn a total count across a cooking sequence into a requirement that all
   objects appear simultaneously in one frame.
6. Set needs_motion=true when temporal order or a brief action cannot be verified
   reliably from one still image.
7. Set needs_text_reading=true when an exact_text criterion is required, or when a
   decisive count/visual clue is tiny text such as a traffic-light countdown.
8. Populate events when the query explicitly labels multiple events such as E1, E2,
   E3, asks for a separate keyframe for each event, or is a QA narrative whose distinct
   scenes identify one source video before a separate answer is located. A non-QA
   narrative description of several moments used to identify one clip remains KIS:
   keep its criteria in the global criteria list and assign window/video evidence_scope
   as needed. Every event needs a faithful English visual query and globally unique
   criterion IDs such as E1_c1. Set sequence_required=true when their order matters.
9. Set task_type="trake" only when the query asks for multiple event keyframes in
   temporal order (especially labeled E1, E2, ...). Set task_type="kis" when the
   output is one representative keyframe, including when the description narrates
   several moments from the source clip.
10. Set task_type="qa" only when the query asks a question requiring an answer, for
    example a name, number, identity, location, or object type. For QA, extract the
    question into qa_question, name the visual object/evidence to locate in
    target_object using a concise English CLIP-friendly phrase (for example an address
    sign, title card, product label or the object whose identity is asked), and set
    answer_expected_visible=true only when the answer should
    be readable or visually inferable from frames. Never assume speech is available.
    For non-QA set qa_question and target_object to empty strings and
    answer_expected_visible=false.
11. For QA set qa_mode to exactly one of:
    - single_frame: scene identity and answer can normally be established in one image;
    - short_window: nearby frames are needed for an action, count or brief transition;
    - long_range_temporal: two or more distinct scenes may be far apart in one video,
      or the story scenes identify the video while the answer must be found elsewhere
      in that video's timeline (for example on another sign or title card).
    For non-QA set qa_mode="not_applicable". Do not choose long_range_temporal merely
    because a question is verbose; require distinct moments or separate answer evidence.
Do not answer the query. Only prepare the visual search plan.
""".strip()


def contact_rank_prompt(
    query: str,
    plan: QueryPlan,
    candidates: list[Candidate],
    limit: int,
) -> str:
    manifest = [
        {
            "candidate_id": item.candidate_id,
            "video_id": item.video_id,
            "keyframe_ordinal": item.ordinal,
            "frame_idx": item.frame_idx,
            "pts_time": item.pts_time,
            "retrieval_score": round(item.retrieval_score, 6),
        }
        for item in candidates
    ]
    return f"""
You are comparing contact sheets of retrieval candidates for an exact video keyframe.

Original query:
{query}

Mandatory visual conditions:
{json.dumps(plan.must_have, ensure_ascii=False)}

Candidate manifest:
{json.dumps(manifest, ensure_ascii=False)}

Inspect the pixels, not retrieval scores. Compare candidates against each other.
Penalize a candidate strongly if it misses even one mandatory subject, attribute,
object, action, spatial relation, visible text, count, or temporal clue. Select at
most {limit} candidate IDs, best first. candidate_id must exactly match a label in
the sheets. Mark sufficient_match only when at least one candidate is plausibly an
exact match. If none is sufficient, write one complete refined English retrieval
query emphasizing the discriminative details that are missing; it must still
preserve the full composition of the original query.
""".strip()


def sample_select_prompt(
    query: str, plan: QueryPlan, samples: list[SampleFrame]
) -> str:
    manifest = [sample.to_dict() for sample in samples]
    return f"""
Find the moment that best represents the requested action or temporal event.

Original query:
{query}

Mandatory conditions:
{json.dumps(plan.must_have, ensure_ascii=False)}

Sample manifest:
{json.dumps(manifest, ensure_ascii=False)}

The sheets show time-ordered sampled frames. Select exactly one visible sample ID
only if it is supported by the pixels. Prefer the clearest moment satisfying the
whole query, not merely the closest timestamp. Return null if the event is absent.
""".strip()


def video_rank_prompt(
    query: str,
    plan: QueryPlan,
    representatives: list[Candidate],
    limit: int,
) -> str:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in representatives:
        grouped.setdefault(item.video_id, []).append(
            {
                "candidate_id": item.candidate_id,
                "keyframe_ordinal": item.ordinal,
                "frame_idx": item.frame_idx,
                "pts_time": item.pts_time,
            }
        )
    return f"""
The normal keyframe shortlist was not verified. Inspect these uniformly sampled
representatives from candidate videos and choose videos worth a local keyframe scan.

Original query (still authoritative for final verification):
{query}

Mandatory visual conditions:
{json.dumps(plan.must_have, ensure_ascii=False)}

Video manifest, each containing sampled candidate IDs visible in the sheets:
{json.dumps(grouped, ensure_ascii=False)}

Do not choose a video merely because one generic object matches. Prefer a video that
plausibly contains the full scene, people, objects and composition, even if decisive
small text is unreadable in this coarse sample. Select at most {limit} distinct
video_id values from the manifest, best first. This step is only a recall-oriented
video choice; it is not an exact-keyframe verification.
""".strip()


def qa_video_screen_prompt(
    query: str,
    plan: QueryPlan,
    candidates: list[Candidate],
    limit: int,
) -> str:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in candidates:
        grouped.setdefault(item.video_id, []).append(
            {
                "candidate_id": item.candidate_id,
                "keyframe_ordinal": item.ordinal,
                "frame_idx": item.frame_idx,
                "pts_time": item.pts_time,
            }
        )
    criteria = [
        {
            "criterion_id": item.criterion_id,
            "description": item.description,
            "criterion_type": item.criterion_type,
            "evidence_scope": item.evidence_scope,
        }
        for item in plan.criteria
        if item.required and item.criterion_type != "context"
    ]
    events = [
        {
            "event_id": event.event_id,
            "description_vi": event.description_vi,
            "visual_query_en": event.visual_query_en,
        }
        for event in plan.events
    ]
    return f"""
Screen a broad QA video shortlist using sparse CLIP evidence frames.

Original query:
{query}

Question:
{plan.qa_question}

QA mode: {plan.qa_mode}

Story events:
{json.dumps(events, ensure_ascii=False)}

Required criteria:
{json.dumps(criteria, ensure_ascii=False)}

Video manifest:
{json.dumps(grouped, ensure_ascii=False)}

This is a recall-oriented screening stage, not final verification. Frames for one
video may come from different times because the requested evidence can be distributed
throughout a sequence. Prefer videos that collectively show several distinctive
criteria. Do not let one easy generic frame, such as merely showing fish or a kitchen,
outweigh missing actions, ingredients, counts, or text. Select at most {limit}
distinct video_id values, best first. A coarse frame may be partial, so retain a video
when multiple cards together plausibly support the complete query.
For long_range_temporal, the answer may be in a third moment that is not one of the
story events. Keep a video when its sparse cards plausibly match either the complete
story, multiple events, or the answer-locator scene; final verification will inspect
that video's timeline more densely.
""".strip()


def final_select_prompt(
    query: str, plan: QueryPlan, candidates: list[Candidate], limit: int
) -> str:
    manifest = [
        {
            "candidate_id": item.candidate_id,
            "video_id": item.video_id,
            "keyframe_ordinal": item.ordinal,
            "frame_idx": item.frame_idx,
            "pts_time": item.pts_time,
        }
        for item in candidates
    ]
    criteria = [
        {
            "criterion_id": item.criterion_id,
            "criterion_type": item.criterion_type,
            "description": item.description,
            "value": item.value,
            "required": item.required,
            "classification_confidence": item.classification_confidence,
        }
        for item in plan.criteria
    ]
    return f"""
Choose the exact final keyframe from neighboring full-scene candidates.

Original query:
{query}

Mandatory visual conditions:
{json.dumps(plan.must_have, ensure_ascii=False)}

Typed criteria:
{json.dumps(criteria, ensure_ascii=False)}

Candidate manifest:
{json.dumps(manifest, ensure_ascii=False)}

Use only visible evidence and compare all candidates jointly. Return up to {limit}
different candidates in ranked_candidates, best first. If at least five candidates
are present in the manifest, rank at least five. A ranked row may have verified=false;
do not call a merely similar frame exact just to fill the list. Give every row its
own confidence, satisfied conditions, missing conditions and concise reason.
For the top-level selection and every ranked row, return one criterion_results entry
for every typed criterion using its exact criterion_id. A context criterion does not
need to be visible; assess whether the frame is visually compatible with it, but do
not fail a frame solely because the place/event name is not written on screen.

selected_candidate_id must be the first and best defensible exact match. Choose the
frame that shows the mandatory conditions most clearly while preserving the requested
composition. The top-level confidence/evidence describes that selected frame. Do not
invent text or tiny details. If no candidate is defensible, set verified=false and
selected_candidate_id=null, but still return the strongest ranked candidates for
human inspection.
""".strip()


def local_anchor_prompt(
    query: str,
    plan: QueryPlan,
    candidates: list[Candidate],
    limit: int,
) -> str:
    manifest = [
        {
            "candidate_id": item.candidate_id,
            "video_id": item.video_id,
            "keyframe_ordinal": item.ordinal,
            "frame_idx": item.frame_idx,
            "pts_time": item.pts_time,
        }
        for item in candidates
    ]
    return f"""
Choose temporal anchors for a local keyframe scan within one candidate video.

Original query:
{query}

Mandatory visual conditions:
{json.dumps(plan.must_have, ensure_ascii=False)}

Candidate manifest:
{json.dumps(manifest, ensure_ascii=False)}

This is a recall step, not final verification. Select at most {limit} visible
candidate IDs whose surrounding time windows are most worth inspecting. A selected
anchor may miss small text or a transient action; prefer the frame that is closest
to the complete scene. Do not select a generic frame from an unrelated scene.
""".strip()


def rescue_zoom_prompt(
    query: str, plan: QueryPlan, candidates: list[Candidate], limit: int
) -> str:
    manifest = [
        {
            "candidate_id": item.candidate_id,
            "video_id": item.video_id,
            "keyframe_ordinal": item.ordinal,
            "frame_idx": item.frame_idx,
            "pts_time": item.pts_time,
        }
        for item in candidates
    ]
    criteria = [
        {
            "criterion_id": item.criterion_id,
            "criterion_type": item.criterion_type,
            "description": item.description,
            "value": item.value,
            "required": item.required,
            "classification_confidence": item.classification_confidence,
        }
        for item in plan.criteria
    ]
    return f"""
Perform a high-detail rescue verification for exact video keyframe selection.

Original query:
{query}

Mandatory visual conditions:
{json.dumps(plan.must_have, ensure_ascii=False)}

Typed criteria:
{json.dumps(criteria, ensure_ascii=False)}

Candidate manifest:
{json.dumps(manifest, ensure_ascii=False)}

The contact sheets first show full frames. They may then show labeled detail crops.
Every crop is from the same candidate_id named on its card, and may only magnify
evidence for that same candidate. Never combine evidence from different candidate
IDs. Return up to {limit} different candidates in ranked_candidates, best first; if
at least five candidates are available, rank at least five. Give each row its own
verified status, confidence and evidence. Similar alternatives may be listed with
verified=false, but must not be promoted to exact matches merely to fill the list.
For the top-level selection and every ranked row, return one criterion_results entry
for every typed criterion using its exact criterion_id. Context criteria are retrieval
clues and do not require their names to be readable on screen.
selected_candidate_id is the first and best defensible exact match. If no one
candidate is defensible, return null while retaining the ranked list for inspection.
""".strip()


def multi_event_select_prompt(
    query: str,
    plan: QueryPlan,
    video_id: str,
    candidates: list[Candidate],
    limit: int,
) -> str:
    manifest = [
        {
            "candidate_id": item.candidate_id,
            "keyframe_ordinal": item.ordinal,
            "frame_idx": item.frame_idx,
            "pts_time": item.pts_time,
        }
        for item in sorted(candidates, key=lambda candidate: candidate.ordinal)
    ]
    events = [
        {
            "event_id": event.event_id,
            "description_vi": event.description_vi,
            "visual_query_en": event.visual_query_en,
            "criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "criterion_type": criterion.criterion_type,
                    "description": criterion.description,
                    "value": criterion.value,
                    "required": criterion.required,
                    "classification_confidence": criterion.classification_confidence,
                }
                for criterion in event.criteria
            ],
        }
        for event in plan.events
    ]
    return f"""
Verify a multi-event TRAKE sequence inside one candidate video.

Original query:
{query}

Candidate video: {video_id}
Ordered events:
{json.dumps(events, ensure_ascii=False)}

Time-ordered candidate manifest:
{json.dumps(manifest, ensure_ascii=False)}

The contact sheets show frames from this one video and their exact candidate labels.
For every requested event, select the clearest frame depicting that event and return
up to {limit} ranked alternatives, best first. Always return the strongest candidates
visible in the manifest for human review, even when none is an exact match. If at
least five candidates are available, rank at least five. Mark weak or merely similar
alternatives verified=false; never make them exact merely to fill the list.
Return criterion_results for every criterion of that event using exact criterion IDs.

The selected E1, E2, ... frames must occur in strictly increasing pts_time order.
Do not replace a requested fruit/object with a generic orchard or a table containing
mixed fruits. A wide context frame is insufficient when the event asks for the first
scene containing one particular fruit. Set an event verified=false when its specific
subject is not visibly defensible. Set overall verified=true only when every event is
verified in this same video and the selected times follow the requested order.
""".strip()


def qa_evidence_prompt(
    query: str, plan: QueryPlan, candidates: list[Candidate], result_count: int
) -> str:
    manifest = [
        {
            "candidate_id": candidate.candidate_id,
            "video_id": candidate.video_id,
            "keyframe_ordinal": candidate.ordinal,
            "frame_idx": candidate.frame_idx,
            "pts_time": candidate.pts_time,
        }
        for candidate in candidates
    ]
    criteria = [
        {
            "criterion_id": criterion.criterion_id,
            "criterion_type": criterion.criterion_type,
            "description": criterion.description,
            "value": criterion.value,
            "required": criterion.required,
            "evidence_scope": criterion.evidence_scope,
        }
        for criterion in plan.criteria
    ]
    events = [
        {
            "event_id": event.event_id,
            "description_vi": event.description_vi,
            "visual_query_en": event.visual_query_en,
        }
        for event in plan.events
    ]
    return f"""
Perform visual QA evidence verification across sampled frames from several videos.

Original query:
{query}

Question to answer:
{plan.qa_question}

Target object/evidence to locate:
{plan.target_object}

QA mode:
{plan.qa_mode}

Story events used to identify the same source video:
{json.dumps(events, ensure_ascii=False)}

Criteria:
{json.dumps(criteria, ensure_ascii=False)}

Candidate manifest:
{json.dumps(manifest, ensure_ascii=False)}

Return exactly {result_count} results from distinct video_id values, best first.
Every result must cite one to six exact evidence_candidate_ids from that video's
visible cards. Always return five reviewable candidates even if none is verified.

For long_range_temporal, first verify that the distinct story events belong to the
same video, even when they are far apart in time. Then search other sampled frames of
that same video for the requested answer. Do not require the answer to appear in the
same frame as a story event. Put only the frame IDs that visibly contain or directly
support the answer in answer_evidence_candidate_ids. For other modes use that field
the same way; return an empty list when the answer is not visible.

Evaluate criteria by evidence_scope: frame must be visible in one cited frame; window
may use an ordered group of cited nearby frames; video may aggregate across cited
frames. For a total object count, use count_status=same_frame only when all objects
are visible together. Use across_frames only when the cited frames defensibly show
distinct objects/steps, not repeated views of the same one. Otherwise use
not_verified. Do not claim an answer that is not readable or visually defensible;
then set answer=null and answer_source=not_visible. Do not rely on audio or speech.
Return criterion_results for every criterion using exact criterion IDs.
""".strip()


def temporal_kis_select_prompt(
    query: str,
    plan: QueryPlan,
    candidates: list[Candidate],
    result_count: int,
) -> str:
    manifest = [
        {
            "candidate_id": candidate.candidate_id,
            "video_id": candidate.video_id,
            "keyframe_ordinal": candidate.ordinal,
            "frame_idx": candidate.frame_idx,
            "pts_time": candidate.pts_time,
        }
        for candidate in candidates
    ]
    criteria = [
        {
            "criterion_id": criterion.criterion_id,
            "criterion_type": criterion.criterion_type,
            "description": criterion.description,
            "required": criterion.required,
            "evidence_scope": criterion.evidence_scope,
        }
        for criterion in plan.criteria
    ]
    return f"""
Verify a temporal KIS description and choose one representative keyframe.

Original query:
{query}

Typed criteria:
{json.dumps(criteria, ensure_ascii=False)}

Candidate manifest:
{json.dumps(manifest, ensure_ascii=False)}

The query may describe several moments from one source video even though the final
KIS answer is one keyframe. Select exactly one source video. All criterion evidence
must come from that same video. Evaluate evidence_scope as follows:
- frame: one cited frame must visibly establish the criterion;
- window: one or more nearby cited frames may establish an action;
- video: evidence may be distributed across the selected video's timeline.

For a temporal criterion, list evidence_candidate_ids in chronological order and do
not claim an order that the timestamps contradict. Do not require every detail to be
visible in selected_candidate_id. That frame should instead be the clearest and most
distinctive requested moment from the verified video. Prefer a frame containing
several important details when available.

Return up to {result_count} ranked candidate keyframes from the selected video, best
first. Return at least five when five valid candidates are available. The first row
must be selected_candidate_id. Set verified=true only when every required criterion
is defensibly supported in the selected video and the representative frame belongs
to it. If no video is fully supported, return the closest video with verified=false,
retain reviewable ranked candidates, and explain what is missing.
""".strip()
