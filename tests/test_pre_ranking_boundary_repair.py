import json
import unittest

from shorts_generator.highlights import (
    _apply_pre_ranking_boundary_repair,
    _eligible_for_pre_ranking_repair,
    repair_boundary_failures_before_ranking,
)


def strong_candidate(start, end, **overrides):
    item = {
        "candidate_id": "candidate_003",
        "title": "Strong candidate",
        "start_time": start,
        "end_time": end,
        "actual_opening_quote": "Opening",
        "actual_closing_quote": "Closing",
        "opening_complete": True,
        "ending_complete": False,
        "standalone_context": True,
        "has_payoff": True,
        "publishable": False,
        "final_score": 75.5,
        "score": 75.5,
        "scores": {
            "standalone_clarity": 8,
            "niche_relevance": 8,
            "payoff_strength": 8,
        },
    }
    item.update(overrides)
    return item


def clean_repair(start, end, **overrides):
    repair = {
        "candidate_id": "candidate_003",
        "proposed_start_time": start,
        "proposed_end_time": end,
        "opening_complete": True,
        "ending_complete": True,
        "standalone_context": True,
        "has_payoff": True,
        "publishable": True,
        "reason": "The adjacent segment completes the thought.",
    }
    repair.update(overrides)
    return repair


class PreRankingBoundaryRepairTests(unittest.TestCase):
    def test_gold_ending_is_repaired_and_revalidated_before_ranking(self):
        segments = [
            {"start": 1470.72, "end": 1490.28, "text": "Gold miners have unusual economics."},
            {"start": 1490.28, "end": 1522.64, "text": "The production spread creates upside, but the risk is"},
            {"start": 1522.64, "end": 1527.28, "text": "limited compared with that potential payoff."},
        ]
        original = strong_candidate(1470.72, 1522.64)
        repaired = _apply_pre_ranking_boundary_repair(
            original, clean_repair(1470.72, 1527.28), segments
        )
        self.assertEqual(1527.28, repaired["end_time"])
        self.assertEqual("limited compared with that potential payoff.", repaired["actual_closing_quote"])
        self.assertTrue(repaired["publishable"])
        self.assertEqual(75.5, repaired["final_score"])

    def test_ai_agents_opening_expands_backward_to_real_boundary(self):
        segments = [
            {"start": 1812.62, "end": 1817.62, "text": "One of the possible innovations is that"},
            {"start": 1817.62, "end": 1840.0, "text": "people will continue building AI agents."},
            {"start": 1840.0, "end": 1874.74, "text": "Those agents may need crypto wallets."},
        ]
        original = strong_candidate(
            1817.62, 1874.74, opening_complete=False, ending_complete=True
        )
        repaired = _apply_pre_ranking_boundary_repair(
            original, clean_repair(1812.62, 1874.74), segments
        )
        self.assertEqual(1812.62, repaired["start_time"])
        self.assertEqual("One of the possible innovations is that", repaired["actual_opening_quote"])

    def test_copycat_and_bitcoin_are_not_eligible_for_rescue(self):
        copycat = strong_candidate(
            100.0, 130.0, opening_complete=False, standalone_context=False
        )
        bitcoin = strong_candidate(
            200.0, 230.0, opening_complete=False, ending_complete=False,
            standalone_context=False, has_payoff=False,
        )
        self.assertFalse(_eligible_for_pre_ranking_repair(copycat))
        self.assertFalse(_eligible_for_pre_ranking_repair(bitcoin))

    def test_clean_latvia_candidate_skips_pre_ranking_repair(self):
        latvia = strong_candidate(
            1565.16, 1622.74, ending_complete=True, publishable=True
        )
        called = False

        def fail_if_called(prompt):
            nonlocal called
            called = True
            raise AssertionError("clean candidate must not invoke repair")

        result = repair_boundary_failures_before_ranking([latvia], [], fail_if_called)
        self.assertFalse(called)
        self.assertEqual((1565.16, 1622.74), (result[0]["start_time"], result[0]["end_time"]))

    def test_invalid_timestamp_rejects_repair_without_snapping(self):
        segments = [
            {"start": 1470.72, "end": 1522.64, "text": "The selected claim continues."},
            {"start": 1522.64, "end": 1527.28, "text": "Here is its completion."},
        ]
        repaired = _apply_pre_ranking_boundary_repair(
            strong_candidate(1470.72, 1522.64),
            clean_repair(1470.72, 1527.281),
            segments,
        )
        self.assertIsNone(repaired)

    def test_repair_must_revalidate_every_hard_gate(self):
        segments = [
            {"start": 1470.72, "end": 1522.64, "text": "The selected claim continues."},
            {"start": 1522.64, "end": 1527.28, "text": "Here is its completion."},
        ]
        repaired = _apply_pre_ranking_boundary_repair(
            strong_candidate(1470.72, 1522.64),
            clean_repair(1470.72, 1527.28, standalone_context=False),
            segments,
        )
        self.assertIsNone(repaired)

    def test_eligible_candidates_use_one_batched_call_with_local_context(self):
        segments = [
            {"start": 1465.0, "end": 1470.72, "text": "Previous setup."},
            {"start": 1470.72, "end": 1522.64, "text": "Selected gold thought."},
            {"start": 1522.64, "end": 1527.28, "text": "Immediate completion."},
        ]
        prompts = []

        def fake_llm(prompt):
            prompts.append(prompt)
            return json.dumps({"repairs": [clean_repair(1470.72, 1527.28)]})

        result = repair_boundary_failures_before_ranking(
            [strong_candidate(1470.72, 1522.64)], segments, fake_llm
        )
        self.assertEqual(1, len(prompts))
        self.assertIn("Immediate completion.", prompts[0])
        self.assertNotIn("final semantic-boundary critic", prompts[0])
        self.assertTrue(result[0]["publishable"])


if __name__ == "__main__":
    unittest.main()
