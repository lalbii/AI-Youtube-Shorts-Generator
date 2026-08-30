"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - high-recall candidate discovery
  - exact-range publishability judging
  - deterministic component scoring and overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive either MuAPI (default, --mode api) or a direct local LLM client
(--mode local).
"""
import json
import math
import re
from typing import Callable, Dict, List, Optional

from . import muapi


LLMFn = Callable[[str], str]


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are a high-recall short-form video candidate scout.

{virality_criteria}

Content type: {content_type} | Density: {density}

Your task: discover promising candidate passages for a separate editor to judge later.

Rules:
- Return approximately {candidate_count} candidates when the transcript supports them
- Favor recall and variety: include concrete numbers, strong claims, useful explanations, business stories, predictions, tension, and payoffs
- Each start_time and end_time must exactly match bracketed transcript segment boundaries for one continuous passage
- Duration must be 20-60 seconds; strongly prefer 25-45 seconds
- Prefer complete passages, but do not hide uncertainty by inventing spoken lines
- display_hook is optional generated overlay copy and does not need to be verbatim
- provisional_score is for discovery diagnostics only and will never determine finalists
- Do not judge or provide a final overall score

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"provisional_score":int,"display_hook":"string","discovery_reason":"string"}}]}}"""


TARGET_NICHE = "AI × Business × Money"


JUDGE_SYSTEM_PROMPT = """You are the publishability judge for short-form clips in the niche AI × Business × Money.

Judge each EXACT timestamp range from its actual spoken transcript, not from its title, display_hook, discovery reason, or provisional score. A generated display_hook must never influence any score.

Relevant themes include AI changing work or business, automation, founders, startup mistakes, revenue/profit/cost figures, investing and business decisions, contrarian opinions, unusual business stories, business opportunities, concrete predictions, and practical insights.

For each candidate:
- Inspect the selected transcript plus the immediate before/after context.
- opening_complete must be false for a mid-sentence/mid-thought opening, a dangling conjunction, or an unexplained reference.
- ending_complete must be false when the thought or answer continues, the payoff is just after the end, or the clip ends during setup.
- standalone_context must be false if a cold viewer cannot understand the passage.
- has_payoff requires a real explanation, conclusion, surprising fact, practical insight, punchline, or completed claim.
- publishable may be true only when all four checks are true.
- You may repair boundaries using corrected_start_time and corrected_end_time, but both must exactly match supplied transcript segment boundaries, remain one continuous passage, and normally last 20-60 seconds. Expand only for necessary setup/payoff, never as blind padding.
- actual_hook_strength scores only the first ACTUALLY SPOKEN content in the corrected range. Never score display_hook.
- Return all eight component scores from 0 to 10. Do not return or calculate an overall score.
- topic_key should be a short semantic label used for lightweight diversity.

Respond ONLY with valid JSON:
{"judgments":[{"candidate_id":"candidate_000","corrected_start_time":0.0,"corrected_end_time":30.0,"opening_complete":true,"ending_complete":true,"standalone_context":true,"has_payoff":true,"publishable":true,"scores":{"actual_hook_strength":0,"standalone_clarity":0,"payoff_strength":0,"niche_relevance":0,"novelty":0,"practical_value":0,"emotional_tension":0,"quotability_shareability":0},"display_hook":"optional overlay copy","topic_key":"short semantic label","judge_reason":"concise exact-range explanation"}]}"""


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
MIN_HIGHLIGHT_SECONDS = 20
MAX_HIGHLIGHT_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # cap LLM polls at 5 min — a wedged call should fail fast
MAX_HIGHLIGHT_API_ATTEMPTS = 3

SCORE_WEIGHTS = {
    "actual_hook_strength": 0.20,
    "standalone_clarity": 0.18,
    "payoff_strength": 0.16,
    "niche_relevance": 0.15,
    "novelty": 0.10,
    "practical_value": 0.08,
    "emotional_tension": 0.05,
    "quotability_shareability": 0.08,
}


def call_muapi_llm(prompt: str) -> str:
    """Default LLM backend: MuAPI gpt-5-mini."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def _parse_json_loose(raw: str) -> Dict:
    """gpt-5-4 sometimes wraps JSON in markdown fences — strip and parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fit_to_transcript(
    start: float,
    end: float,
    segments: List[Dict],
) -> Optional[tuple]:
    """Align a proposed range to nearby segment boundaries without padding it."""
    ordered = sorted(segments, key=lambda s: float(s["start"]))
    anchor = next(
        (
            s
            for s in ordered
            if float(s["start"]) <= start < float(s["end"])
        ),
        next((s for s in ordered if float(s["start"]) >= start), None),
    )
    if anchor is None:
        return None

    aligned_start = float(anchor["start"])
    possible_ends = [float(s["end"]) for s in ordered if float(s["end"]) > aligned_start]
    if not possible_ends:
        return None
    fitted_end = min(possible_ends, key=lambda value: abs(value - end))
    if not MIN_HIGHLIGHT_SECONDS <= fitted_end - aligned_start <= MAX_HIGHLIGHT_SECONDS:
        return None
    return aligned_start, fitted_end


def _sanitize_highlights(
    raw_highlights: object,
    duration: float,
    segments: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Normalize model output and enforce transcript-aligned clip duration."""
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue
        if segments:
            fitted = _fit_to_transcript(start, end, segments)
            if fitted is None:
                continue
            start, end = fitted
        elif not MIN_HIGHLIGHT_SECONDS <= end - start <= MAX_HIGHLIGHT_SECONDS:
            # Repair requires real transcript boundaries; do not blindly clamp.
            continue


        provisional_score = max(
            0,
            min(
                100,
                _coerce_int(
                    item.get("provisional_score", item.get("score")),
                    default=0,
                ),
            ),
        )
        display_hook = str(
            item.get("display_hook") or item.get("hook_sentence") or ""
        ).strip()
        discovery_reason = str(
            item.get("discovery_reason") or item.get("virality_reason") or ""
        ).strip()
        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip(),
                "start_time": start,
                "end_time": end,
                "provisional_score": provisional_score,
                "display_hook": display_hook,
                # Compatibility aliases are retained, but neither participates
                # in exact-range judging or deterministic final scoring.
                "score": provisional_score,
                "hook_sentence": display_hook,
                "discovery_reason": discovery_reason,
                "virality_reason": discovery_reason,
            }
        )

    return cleaned


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(
        f"[{float(s['start']):.2f}s - {float(s['end']):.2f}s] {str(s['text']).strip()}"
        for s in segments
    )


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        window_end = min(end + CHUNK_OVERLAP_SECONDS, duration)
        chunk_segs = [
            {
                **s,
                "start": float(s["start"]) - start,
                "end": float(s["end"]) - start,
            }
            for s in segments
            if s["start"] >= start and s["end"] <= window_end
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = window_end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
    transcript_segments: Optional[List[Dict]] = None,
) -> Dict:
    # num_clips is the requested discovery count here, not the final quota.
    natural_max = max(2 if is_chunk else 3, int(duration / 45))
    candidate_count = max(1, min(num_clips, natural_max))
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        candidate_count=candidate_count,
    )
    base_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            highlights = _sanitize_highlights(
                parsed.get("highlights"),
                duration=duration,
                segments=transcript_segments,
            )
            if highlights:
                return {"highlights": highlights}
            last_error = "no valid highlights in response"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(
                f"[highlights] invalid model output on attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS}; retrying",
                flush=True,
            )
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, provisional_score, display_hook, discovery_reason."
                + " No markdown fences, no commentary."
            )

    raise RuntimeError(
        f"Highlight generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )


def calculate_final_score(scores: Dict[str, object]) -> float:
    """Compute the transparent 0-100 score; LLM overall scores are ignored."""
    total = 0.0
    for component, weight in SCORE_WEIGHTS.items():
        value = max(0.0, min(10.0, _coerce_float(scores.get(component), default=0.0)))
        total += value * weight
    return round(total * 10.0, 1)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _format_segments(segments: List[Dict]) -> List[Dict[str, object]]:
    return [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": str(segment.get("text", "")).strip(),
        }
        for segment in segments
    ]


def _candidate_evidence(candidate: Dict, segments: List[Dict]) -> Dict[str, object]:
    """Attach the exact selected transcript and immediate surrounding context."""
    ordered = sorted(segments, key=lambda segment: float(segment["start"]))
    start = float(candidate["start_time"])
    end = float(candidate["end_time"])
    selected_indexes = [
        index
        for index, segment in enumerate(ordered)
        if float(segment["start"]) >= start - 1e-6
        and float(segment["end"]) <= end + 1e-6
    ]
    if not selected_indexes:
        before: List[Dict] = []
        selected: List[Dict] = []
        after: List[Dict] = []
    else:
        first = selected_indexes[0]
        last = selected_indexes[-1]
        before = ordered[max(0, first - 2):first]
        selected = ordered[first:last + 1]
        after = ordered[last + 1:last + 3]

    return {
        "candidate_id": candidate["candidate_id"],
        "title": candidate.get("title", ""),
        "proposed_start_time": start,
        "proposed_end_time": end,
        "target_niche": TARGET_NICHE,
        "context_before": _format_segments(before),
        "selected_transcript": _format_segments(selected),
        "context_after": _format_segments(after),
    }


def call_publishability_judge(
    candidates: List[Dict],
    segments: List[Dict],
    llm_fn: LLMFn,
) -> List[Dict]:
    """Run one independent batch judge call over exact candidate evidence."""
    evidence = [_candidate_evidence(candidate, segments) for candidate in candidates]
    base_prompt = (
        JUDGE_SYSTEM_PROMPT
        + "\n\nCandidates and transcript evidence:\n"
        + json.dumps(evidence, ensure_ascii=False)
    )
    prompt = base_prompt
    last_error = "unknown"
    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            judgments = parsed.get("judgments")
            if isinstance(judgments, list):
                return [item for item in judgments if isinstance(item, dict)]
            last_error = "missing judgments array"
        except Exception as exc:
            last_error = str(exc)
        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            prompt = base_prompt + "\n\nIMPORTANT: Return only valid JSON with a judgments array."
    raise RuntimeError(
        "Publishability judge produced invalid output after "
        f"{MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )


def _match_boundary(value: object, boundaries: List[float]) -> Optional[float]:
    proposed = _coerce_float(value, default=float("nan"))
    if not math.isfinite(proposed) or not boundaries:
        return None
    nearest = min(boundaries, key=lambda boundary: abs(boundary - proposed))
    return nearest if abs(nearest - proposed) <= 0.05 else None


def _obviously_incomplete_opening(text: str) -> bool:
    """Catch only clear dangling references; nuanced cases remain judge-owned."""
    normalized = re.sub(r'^[\s\"\'“”‘’]+', "", text).strip().lower()
    return bool(
        re.match(
            r"^(?:and|but|so)\s+(?:they|he|she|it|this|that|these|those)\b",
            normalized,
        )
    )


def _apply_judgment(candidate: Dict, judgment: Dict, segments: List[Dict]) -> Dict:
    ordered = sorted(segments, key=lambda segment: float(segment["start"]))
    start_boundaries = [float(segment["start"]) for segment in ordered]
    end_boundaries = [float(segment["end"]) for segment in ordered]
    proposed_start = judgment.get("corrected_start_time", candidate["start_time"])
    proposed_end = judgment.get("corrected_end_time", candidate["end_time"])
    start = _match_boundary(proposed_start, start_boundaries)
    end = _match_boundary(proposed_end, end_boundaries)
    boundary_valid = (
        start is not None
        and end is not None
        and MIN_HIGHLIGHT_SECONDS <= end - start <= MAX_HIGHLIGHT_SECONDS
    )

    selected = []
    if boundary_valid:
        selected = [
            segment
            for segment in ordered
            if float(segment["start"]) >= start - 1e-6
            and float(segment["end"]) <= end + 1e-6
        ]
        boundary_valid = bool(selected)

    actual_opening_quote = (
        str(selected[0].get("text", "")).strip() if selected else ""
    )
    actual_closing_quote = (
        str(selected[-1].get("text", "")).strip() if selected else ""
    )
    opening_complete = (
        boundary_valid
        and _coerce_bool(judgment.get("opening_complete"))
        and not _obviously_incomplete_opening(actual_opening_quote)
    )
    ending_complete = boundary_valid and _coerce_bool(judgment.get("ending_complete"))
    standalone_context = boundary_valid and _coerce_bool(judgment.get("standalone_context"))
    has_payoff = boundary_valid and _coerce_bool(judgment.get("has_payoff"))
    publishable = (
        _coerce_bool(judgment.get("publishable"))
        and opening_complete
        and ending_complete
        and standalone_context
        and has_payoff
    )

    raw_scores = judgment.get("scores") if isinstance(judgment.get("scores"), dict) else {}
    scores = {
        component: max(
            0.0,
            min(10.0, _coerce_float(raw_scores.get(component), default=0.0)),
        )
        for component in SCORE_WEIGHTS
    }
    final_score = calculate_final_score(scores)
    display_hook = str(
        judgment.get("display_hook") or candidate.get("display_hook") or ""
    ).strip()
    judge_reason = str(judgment.get("judge_reason") or "").strip()

    result = {
        **candidate,
        "start_time": start if start is not None else float(candidate["start_time"]),
        "end_time": end if end is not None else float(candidate["end_time"]),
        "actual_opening_quote": actual_opening_quote,
        "actual_closing_quote": actual_closing_quote,
        "display_hook": display_hook,
        "hook_sentence": display_hook,
        "opening_complete": bool(opening_complete),
        "ending_complete": bool(ending_complete),
        "standalone_context": bool(standalone_context),
        "has_payoff": bool(has_payoff),
        "publishable": bool(publishable),
        "scores": scores,
        "final_score": final_score,
        # score/virality_reason remain compatibility aliases for renderers/UI.
        "score": final_score,
        "judge_reason": judge_reason,
        "virality_reason": judge_reason,
        "topic_key": str(judgment.get("topic_key") or "").strip(),
    }
    return result


def judge_candidates(
    candidates: List[Dict],
    segments: List[Dict],
    llm_fn: LLMFn,
) -> List[Dict]:
    identified = [
        {**candidate, "candidate_id": f"candidate_{index:03d}"}
        for index, candidate in enumerate(candidates)
    ]
    judgments = call_publishability_judge(identified, segments, llm_fn)
    by_id = {
        str(judgment.get("candidate_id")): judgment
        for judgment in judgments
        if judgment.get("candidate_id") is not None
    }
    return [
        _apply_judgment(candidate, by_id.get(candidate["candidate_id"], {}), segments)
        for candidate in identified
    ]


def _ranking_score(highlight: Dict) -> float:
    return _coerce_float(
        highlight.get("final_score", highlight.get("score")),
        default=0.0,
    )


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=_ranking_score, reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def _normalized_topic(highlight: Dict) -> str:
    topic = str(highlight.get("topic_key") or highlight.get("title") or "").lower()
    return " ".join(re.findall(r"[a-z0-9]+", topic))


def rank_publishable(highlights: List[Dict], num_clips: int) -> List[Dict]:
    """Apply hard gates, overlap dedupe, then lightweight topic diversity."""
    ranked = dedupe_highlights([item for item in highlights if item.get("publishable")])
    if num_clips <= 0:
        return []

    selected: List[Dict] = []
    deferred: List[Dict] = []
    used_topics = set()
    for highlight in ranked:
        topic = _normalized_topic(highlight)
        if topic and topic in used_topics:
            deferred.append(highlight)
            continue
        selected.append(highlight)
        if topic:
            used_topics.add(topic)
        if len(selected) == num_clips:
            return selected

    # Reuse a topic only when there are not enough diverse publishable choices.
    for highlight in deferred:
        selected.append(highlight)
        if len(selected) == num_clips:
            break
    return selected


FINAL_BOUNDARY_CRITIC_PROMPT = """You are the final semantic-boundary critic for already-selected AI × Business × Money finalists. Check only whether the start is a clean standalone opening and the end is semantically complete. Never rescore, rerank, or reconsider content quality. Repair only with timestamps present in the supplied local evidence. Expand backward only for a mid-sentence, mid-thought, or context-dependent start; expand forward only for an obvious completion/payoff. Prefer the tightest complete clip and add no unnecessary context or new topic. Preserve strong standalone hooks even when earlier context exists. If clean, keep timestamps and set needs_repair false. If broken and not locally repairable, set needs_repair true; it will be rejected. Return JSON only:
{"reviews":[{"candidate_id":"candidate_000","start_ok":true,"end_ok":true,"needs_repair":false,"proposed_start_time":0.0,"proposed_end_time":30.0,"reason":"boundary-only explanation"}]}"""


def _final_boundary_evidence(item, segments):
    ordered = sorted(segments, key=lambda s: float(s["start"]))
    start, end = float(item["start_time"]), float(item["end_time"])
    indexes = [i for i, s in enumerate(ordered) if float(s["start"]) >= start - 1e-6 and float(s["end"]) <= end + 1e-6]
    def at(i):
        return _format_segments([ordered[i]])[0] if 0 <= i < len(ordered) else None
    if indexes:
        first, last = indexes[0], indexes[-1]
        local = (at(first - 1), at(first), at(first + 1) if first < last else None, at(last), at(last + 1))
    else:
        local = (None,) * 5
    return dict(zip(("previous_segment", "first_selected_segment", "second_selected_segment", "last_selected_segment", "next_segment"), local), candidate_id=item["candidate_id"], title=item.get("title", ""), target_niche=TARGET_NICHE, proposed_start_time=start, proposed_end_time=end)


def call_final_boundary_critic(finalists, segments, llm_fn):
    evidence = [_final_boundary_evidence(item, segments) for item in finalists]
    base = FINAL_BOUNDARY_CRITIC_PROMPT + "\n\nFinalists and local transcript evidence:\n" + json.dumps(evidence, ensure_ascii=False)
    prompt, error = base, "unknown"
    for attempt in range(MAX_HIGHLIGHT_API_ATTEMPTS):
        try:
            reviews = _parse_json_loose(llm_fn(prompt)).get("reviews")
            if isinstance(reviews, list):
                return [r for r in reviews if isinstance(r, dict)]
            error = "missing reviews array"
        except Exception as exc:
            error = str(exc)
        prompt = base + "\n\nIMPORTANT: Return only valid JSON with a reviews array."
    raise RuntimeError(f"Final boundary critic produced invalid output: {error}")


def _apply_final_boundary_review(item, review, segments):
    evidence = _final_boundary_evidence(item, segments)
    local = [evidence[k] for k in ("previous_segment", "first_selected_segment", "second_selected_segment", "last_selected_segment", "next_segment") if evidence[k]]
    repair, reason = _coerce_bool(review.get("needs_repair")), str(review.get("reason") or "").strip()
    if _coerce_bool(review.get("start_ok")) and _coerce_bool(review.get("end_ok")) and not repair:
        return {**item, "boundary_repaired": False, "boundary_critic_reason": reason}
    if not repair:
        return None
    start = _match_boundary(review.get("proposed_start_time"), [float(s["start"]) for s in local])
    end = _match_boundary(review.get("proposed_end_time"), [float(s["end"]) for s in local])
    if start is None or end is None or end <= start:
        return None
    selected = [s for s in sorted(segments, key=lambda x: float(x["start"])) if float(s["start"]) >= start - 1e-6 and float(s["end"]) <= end + 1e-6]
    if not selected:
        return None
    return {**item, "start_time": start, "end_time": end, "actual_opening_quote": str(selected[0].get("text", "")).strip(), "actual_closing_quote": str(selected[-1].get("text", "")).strip(), "boundary_repaired": start != float(item["start_time"]) or end != float(item["end_time"]), "boundary_critic_reason": reason}


def verify_final_boundaries(finalists, segments, llm_fn):
    if not finalists:
        return []
    reviews = call_final_boundary_critic(finalists, segments, llm_fn)
    by_id = {str(r.get("candidate_id")): r for r in reviews if r.get("candidate_id") is not None}
    results = []
    for item in finalists:
        review = by_id.get(str(item.get("candidate_id")))
        verified = _apply_final_boundary_review(item, review, segments) if review else None
        if verified is not None:
            results.append(verified)
    return results


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Discover broadly, judge exact ranges, and return publishable finalists.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable.
    """
    llm_fn = llm_fn or call_muapi_llm
    duration = transcript.get("duration", 0)
    segments = transcript.get("segments", [])
    candidate_target = min(15, max(10, num_clips * 4))
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        candidates: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            per_chunk_target = max(1, math.ceil(candidate_target / max(1, len(chunks))))
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            result = call_highlight_api(
                text,
                content_info,
                chunk["duration"],
                num_clips=per_chunk_target,
                is_chunk=True,
                llm_fn=llm_fn,
                transcript_segments=chunk["segments"],
            )
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                candidates.append(h)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(
            text,
            content_info,
            duration,
            num_clips=candidate_target,
            llm_fn=llm_fn,
            transcript_segments=segments,
        )
        candidates = result.get("highlights", [])

    if not candidates:
        return {"highlights": [], "candidates": []}

    judged_candidates = judge_candidates(candidates, segments, llm_fn)
    finalists = rank_publishable(judged_candidates, num_clips=num_clips)
    highlights = verify_final_boundaries(finalists, segments, llm_fn)
    return {
        "highlights": highlights,
        "candidates": judged_candidates,
    }
