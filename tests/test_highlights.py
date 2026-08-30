from __future__ import annotations

import json
import unittest

from shorts_generator.highlights import (
    _apply_judgment,
    _sanitize_highlights,
    calculate_final_score,
    get_highlights,
    rank_publishable,
)


SEGMENTS = [
    {"start": float(i), "end": float(i + 10), "text": f"Thought {i}."}
    for i in range(0, 100, 10)
]


def candidate(start: float, end: float, **overrides):
    return {
        "title": "Candidate",
        "start_time": start,
        "end_time": end,
        "provisional_score": 99,
        "score": 99,
        "display_hook": "Generated marketing hook",
        "hook_sentence": "Generated marketing hook",
        "discovery_reason": "Promising topic",
        "virality_reason": "Promising topic",
        **overrides,
    }


def scores(**overrides):
    values = {name: 8 for name in (
        "actual_hook_strength", "standalone_clarity", "payoff_strength",
        "niche_relevance", "novelty", "practical_value",
        "emotional_tension", "quotability_shareability",
    )}
    values.update(overrides)
    return values


def judgment(start: float, end: float, **overrides):
    value = {
        "corrected_start_time": start,
        "corrected_end_time": end,
        "opening_complete": True,
        "ending_complete": True,
        "standalone_context": True,
        "has_payoff": True,
        "publishable": True,
        "scores": scores(),
        "display_hook": "Generated marketing hook",
        "topic_key": "business insight",
        "judge_reason": "Complete standalone passage.",
    }
    value.update(overrides)
    return value


class CandidateBoundaryTests(unittest.TestCase):
    def test_long_range_is_not_blindly_truncated(self):
        result = _sanitize_highlights(
            [{"title": "long", "start_time": 0, "end_time": 95, "score": 90}],
            duration=100,
            segments=SEGMENTS,
        )
        self.assertEqual([], result)

    def test_short_range_is_not_blindly_padded(self):
        result = _sanitize_highlights(
            [{"title": "short", "start_time": 10, "end_time": 14, "score": 90}],
            duration=100,
            segments=SEGMENTS,
        )
        self.assertEqual([], result)


class SelectorRegressionTests(unittest.TestCase):
    def test_mid_thought_opening_and_missing_payoff_are_not_publishable(self):
        segments = [
            {"start": 100.0, "end": 108.56, "text": "They copy ideas from elsewhere."},
            {"start": 108.56, "end": 118.0, "text": "and they are not inventing anything..."},
            {"start": 118.0, "end": 128.56, "text": "They are applying an old pattern."},
            {"start": 128.56, "end": 132.0, "text": "is nothing new."},
        ]
        judged = _apply_judgment(
            candidate(108.56, 128.56),
            judgment(
                108.56, 128.56, ending_complete=False, publishable=True,
                scores=scores(actual_hook_strength=10, payoff_strength=10),
            ),
            segments,
        )

        self.assertEqual("and they are not inventing anything...", judged["actual_opening_quote"])
        self.assertFalse(judged["opening_complete"])
        self.assertFalse(judged["ending_complete"])
        self.assertFalse(judged["publishable"])

    def test_generated_display_hook_never_becomes_actual_opening_quote(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Revenue doubled after we automated support."},
            {"start": 10.0, "end": 20.0, "text": "That was the practical result."},
        ]
        judged = _apply_judgment(
            candidate(0.0, 20.0),
            judgment(
                0.0, 20.0,
                display_hook="This AI trick instantly doubles every business!",
            ),
            segments,
        )

        self.assertEqual("Revenue doubled after we automated support.", judged["actual_opening_quote"])
        self.assertEqual("This AI trick instantly doubles every business!", judged["display_hook"])
        self.assertNotEqual(judged["actual_opening_quote"], judged["display_hook"])

    def test_clean_numerical_business_clip_can_rank_strongly(self):
        segments = [
            {"start": 1490.28, "end": 1498.0, "text": "They produce an ounce of gold for $1,600 and sell it for $5,000."},
            {"start": 1498.0, "end": 1505.0, "text": "That margin comes after digging and processing it."},
            {"start": 1505.0, "end": 1510.36, "text": "That's a fantastic business."},
        ]
        strong_scores = scores(
            actual_hook_strength=9, standalone_clarity=9, payoff_strength=9,
            niche_relevance=8, novelty=8, practical_value=9,
            emotional_tension=6, quotability_shareability=8,
        )
        judged = _apply_judgment(
            candidate(1490.28, 1510.36, title="Gold mining margins"),
            judgment(1490.28, 1510.36, scores=strong_scores),
            segments,
        )

        self.assertTrue(judged["publishable"])
        self.assertGreater(judged["final_score"], 80)
        self.assertEqual("That's a fantastic business.", judged["actual_closing_quote"])

    def test_fewer_publishable_candidates_returns_fewer_than_requested(self):
        publishable = [
            {
                "title": f"Good {index}", "start_time": index * 30.0,
                "end_time": index * 30.0 + 20.0, "publishable": True,
                "final_score": 90 - index, "topic_key": f"topic {index}",
            }
            for index in range(2)
        ]
        rejected = {
            "title": "Bad", "start_time": 70.0, "end_time": 90.0,
            "publishable": False, "final_score": 100, "topic_key": "bad",
        }

        result = rank_publishable([*publishable, rejected], num_clips=3)

        self.assertEqual(2, len(result))
        self.assertNotIn("Bad", [item["title"] for item in result])

    def test_final_score_is_python_weighted_and_ignores_llm_overall_score(self):
        component_scores = {
            "actual_hook_strength": 10, "standalone_clarity": 9,
            "payoff_strength": 8, "niche_relevance": 7, "novelty": 6,
            "practical_value": 5, "emotional_tension": 4,
            "quotability_shareability": 3,
        }
        segments = [
            {"start": 0.0, "end": 10.0, "text": "A complete opening."},
            {"start": 10.0, "end": 20.0, "text": "A complete payoff."},
        ]
        judged = _apply_judgment(
            candidate(0.0, 20.0),
            judgment(0.0, 20.0, scores=component_scores, overall_score=1),
            segments,
        )

        self.assertEqual(73.9, calculate_final_score(component_scores))
        self.assertEqual(73.9, judged["final_score"])
        self.assertEqual(73.9, judged["score"])


class LongVideoChunkingTests(unittest.TestCase):
    def test_chunk_discovery_is_relative_and_judging_is_global(self):
        transcript = {
            "duration": 1801.0,
            "segments": SEGMENTS
            + [{**s, "start": s["start"] + 1140, "end": s["end"] + 1140} for s in SEGMENTS]
            + [{"start": 1790.0, "end": 1801.0, "text": "ending."}],
        }
        prompts = []

        def fake_llm(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"content_type":"podcast","density":"medium"}'
            if len(prompts) == 2:
                return '{"highlights":[{"title":"first","start_time":10,"end_time":50,"score":80}]}'
            if len(prompts) == 3:
                return '{"highlights":[{"title":"second","start_time":10,"end_time":50,"score":90}]}'
            if len(prompts) == 4:
                return json.dumps({"judgments": [
                    {"candidate_id": "candidate_000", **judgment(10.0, 50.0, topic_key="first")},
                    {"candidate_id": "candidate_001", **judgment(1150.0, 1190.0, topic_key="second")},
                ]})
            return json.dumps({"reviews": [
                {"candidate_id": "candidate_000", "start_ok": True, "end_ok": True, "needs_repair": False, "proposed_start_time": 10.0, "proposed_end_time": 50.0, "reason": "clean"},
                {"candidate_id": "candidate_001", "start_ok": True, "end_ok": True, "needs_repair": False, "proposed_start_time": 1150.0, "proposed_end_time": 1190.0, "reason": "clean"},
            ]})

        result = get_highlights(transcript, num_clips=2, llm_fn=fake_llm)

        self.assertEqual(5, len(prompts))
        self.assertIn("Return approximately 5 candidates", prompts[1])
        self.assertIn("Return approximately 5 candidates", prompts[2])
        self.assertIn("[10.00s - 20.00s] Thought 10.", prompts[2])
        self.assertNotIn("[1150.00s", prompts[2])
        self.assertIn('"proposed_start_time": 1150.0', prompts[3])
        self.assertIn("final semantic-boundary critic", prompts[4])
        self.assertNotIn("Candidates and transcript evidence", prompts[4])
        second = next(item for item in result["candidates"] if item["title"] == "second")
        self.assertEqual(1150.0, second["start_time"])
        self.assertEqual(1190.0, second["end_time"])


if __name__ == "__main__":
    unittest.main()
