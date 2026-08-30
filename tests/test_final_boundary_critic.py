import unittest

from shorts_generator.highlights import (
    _apply_final_boundary_review,
    _final_boundary_evidence,
    verify_final_boundaries,
)


def finalist(start, end):
    return {
        "candidate_id": "candidate_000",
        "title": "Candidate",
        "start_time": start,
        "end_time": end,
        "actual_opening_quote": "original opening",
        "actual_closing_quote": "original closing",
        "final_score": 87.5,
        "scores": {"actual_hook_strength": 9},
        "judge_reason": "Strong topic and payoff.",
    }


def repaired_range_ok():
    return {
        "original_opening_syntactically_dependent": False,
        "original_opening_semantically_standalone": True,
        "proposed_repair_improves_opening": False,
        "proposed_repair_introduces_unrelated_or_fragmentary_material": False,
        "repaired_opening_complete": True,
        "repaired_ending_complete": True,
        "repaired_standalone_context": True,
        "repaired_has_payoff": True,
        "repaired_publishable": True,
    }


def dependent_opening_repair_ok():
    return {
        **repaired_range_ok(),
        "original_opening_syntactically_dependent": True,
        "original_opening_semantically_standalone": False,
        "proposed_repair_improves_opening": True,
    }


class FinalBoundaryCriticTests(unittest.TestCase):
    def test_ai_agents_context_dependent_start_expands_to_real_boundary(self):
        segments = [
            {"start": 1812.62, "end": 1817.6, "text": "One of the possible innovations is that"},
            {"start": 1817.6, "end": 1830.0, "text": "people will continue building AI agents."},
            {"start": 1830.0, "end": 1842.0, "text": "Those agents will need wallets."},
        ]
        review = {"start_ok": False, "end_ok": True, "needs_repair": True,
                  "proposed_start_time": 1812.62, "proposed_end_time": 1842.0,
                  "reason": "The previous segment supplies the setup.", **dependent_opening_repair_ok()}
        repaired = _apply_final_boundary_review(finalist(1817.6, 1842.0), review, segments)
        self.assertEqual(1812.62, repaired["start_time"])
        self.assertEqual("One of the possible innovations is that", repaired["actual_opening_quote"])
        self.assertEqual(87.5, repaired["final_score"])
        self.assertTrue(repaired["boundary_repaired"])
        self.assertTrue(repaired["final_boundary_repaired"])

    def test_latvian_incomplete_ending_expands_to_sentence_completion(self):
        segments = [
            {"start": 1580.0, "end": 1595.0, "text": "Small markets carry concentration risk."},
            {"start": 1595.0, "end": 1608.92, "text": "So if in Latvian stocks, they're just not liquid enough"},
            {"start": 1608.92, "end": 1614.4, "text": "for me to be convinced that I can get out."},
        ]
        review = {"start_ok": True, "end_ok": False, "needs_repair": True,
                  "proposed_start_time": 1580.0, "proposed_end_time": 1614.4,
                  "reason": "The next segment completes the sentence.", **repaired_range_ok()}
        repaired = _apply_final_boundary_review(finalist(1580.0, 1608.92), review, segments)
        self.assertEqual(1614.4, repaired["end_time"])
        self.assertEqual("for me to be convinced that I can get out.", repaired["actual_closing_quote"])

    def test_strong_gold_opening_cannot_be_unnecessarily_expanded(self):
        segments = [
            {"start": 1484.0, "end": 1490.28, "text": "Here is more context."},
            {"start": 1490.28, "end": 1500.0, "text": "Because they can produce an ounce of gold for $1,600 and sell it for $5,000 right now."},
            {"start": 1500.0, "end": 1515.0, "text": "That is an extraordinary margin."},
        ]
        review = {"start_ok": True, "end_ok": True, "needs_repair": True,
                  "proposed_start_time": 1484.0, "proposed_end_time": 1515.0,
                  "reason": "More context exists."}
        verified = _apply_final_boundary_review(
            finalist(1490.28, 1515.0), review, segments
        )
        self.assertEqual((1490.28, 1515.0), (verified["start_time"], verified["end_time"]))
        self.assertFalse(verified["boundary_repaired"])

    def test_already_clean_clip_is_unchanged(self):
        segments = [
            {"start": 10.0, "end": 20.0, "text": "A clean standalone opening."},
            {"start": 20.0, "end": 35.0, "text": "A complete ending."},
        ]
        review = {"start_ok": True, "end_ok": True, "needs_repair": False,
                  "proposed_start_time": 10.0, "proposed_end_time": 35.0,
                  "reason": "Both boundaries are complete."}
        verified = _apply_final_boundary_review(finalist(10.0, 35.0), review, segments)
        self.assertEqual((10.0, 35.0), (verified["start_time"], verified["end_time"]))
        self.assertFalse(verified["boundary_repaired"])

    def test_timestamp_not_in_local_real_boundaries_rejects_repair(self):
        segments = [
            {"start": 10.0, "end": 20.0, "text": "Needed setup."},
            {"start": 20.0, "end": 40.0, "text": "Selected thought."},
        ]
        review = {"start_ok": False, "end_ok": True, "needs_repair": True,
                  "proposed_start_time": 12.0, "proposed_end_time": 40.0,
                  "reason": "Invented timestamp."}
        self.assertIsNone(_apply_final_boundary_review(finalist(20.0, 40.0), review, segments))

    def test_evidence_contains_only_requested_local_segments(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Previous."},
            {"start": 10.0, "end": 20.0, "text": "First."},
            {"start": 20.0, "end": 30.0, "text": "Second."},
            {"start": 30.0, "end": 40.0, "text": "Last."},
            {"start": 40.0, "end": 50.0, "text": "Next."},
        ]
        evidence = _final_boundary_evidence(finalist(10.0, 40.0), segments)
        self.assertEqual("Previous.", evidence["previous_segment"]["text"])
        self.assertEqual("First.", evidence["first_selected_segment"]["text"])
        self.assertEqual("Second.", evidence["second_selected_segment"]["text"])
        self.assertEqual("Last.", evidence["last_selected_segment"]["text"])
        self.assertEqual("Next.", evidence["next_segment"]["text"])
        self.assertNotIn("selected_transcript", evidence)


    def test_ai_agents_repair_is_kept_even_if_needs_repair_flag_is_false(self):
        segments = [
            {"start": 1812.62, "end": 1817.62, "text": "One of the possible innovations is that"},
            {"start": 1817.62, "end": 1830.0, "text": "people will continue building AI agents."},
            {"start": 1830.0, "end": 1863.10, "text": "They could trade without bank accounts."},
        ]
        item = finalist(1817.62, 1863.10)
        item.update({"candidate_id": "candidate_011", "final_score": 80.2})

        def fake_llm(prompt):
            return '''{"reviews":[{"candidate_id":"candidate_011","start_ok":false,"end_ok":true,"needs_repair":false,"original_opening_syntactically_dependent":true,"original_opening_semantically_standalone":false,"proposed_repair_improves_opening":true,"proposed_repair_introduces_unrelated_or_fragmentary_material":false,"proposed_start_time":1812.62,"proposed_end_time":1863.1,"repaired_opening_complete":true,"repaired_ending_complete":true,"repaired_standalone_context":true,"repaired_has_payoff":true,"repaired_publishable":true,"reason":"The previous segment supplies the setup."}]}'''

        highlights = verify_final_boundaries([item], segments, fake_llm)
        self.assertEqual(1, len(highlights))
        self.assertEqual(1812.62, highlights[0]["start_time"])
        self.assertEqual(80.2, highlights[0]["final_score"])
        self.assertTrue(highlights[0]["final_boundary_repaired"])

    def test_clean_gold_and_bitcoin_keep_boundaries(self):
        segments = [
            {"start": 1307.60, "end": 1360.88, "text": "A complete Bitcoin anecdote and conclusion."},
            {"start": 1470.72, "end": 1510.36, "text": "A complete gold margin claim."},
        ]
        cases = [
            (finalist(1470.72, 1510.36), "Opening is standalone and ending is complete."),
            (finalist(1307.60, 1360.88), "Opening is understandable and ending is complete."),
        ]
        for item, reason in cases:
            review = {
                "start_ok": True,
                "end_ok": True,
                "needs_repair": False,
                "proposed_start_time": item["start_time"],
                "proposed_end_time": item["end_time"],
                "reason": reason,
            }
            verified = _apply_final_boundary_review(item, review, segments)
            self.assertEqual(
                (item["start_time"], item["end_time"]),
                (verified["start_time"], verified["end_time"]),
            )
            self.assertFalse(verified["final_boundary_repaired"])

    def test_prior_repair_metadata_survives_clean_final_review(self):
        segments = [{"start": 1307.60, "end": 1360.88, "text": "Complete Bitcoin clip."}]
        item = finalist(1307.60, 1360.88)
        item.update({"boundary_repaired": True, "pre_rank_boundary_repaired": True})
        review = {
            "start_ok": True,
            "end_ok": True,
            "needs_repair": False,
            "proposed_start_time": 1307.60,
            "proposed_end_time": 1360.88,
            "reason": "Already clean.",
        }
        verified = _apply_final_boundary_review(item, review, segments)
        self.assertTrue(verified["boundary_repaired"])
        self.assertTrue(verified["pre_rank_boundary_repaired"])
        self.assertFalse(verified["final_boundary_repaired"])

    def test_failed_revalidation_rejects_otherwise_valid_repair(self):
        segments = [
            {"start": 10.0, "end": 20.0, "text": "Needed setup."},
            {"start": 20.0, "end": 40.0, "text": "Selected thought."},
        ]
        review = {
            "start_ok": False,
            "end_ok": True,
            "needs_repair": True,
            "proposed_start_time": 10.0,
            "proposed_end_time": 40.0,
            **dependent_opening_repair_ok(),
            "repaired_standalone_context": False,
            "reason": "Still depends on outside context.",
        }
        self.assertIsNone(_apply_final_boundary_review(finalist(20.0, 40.0), review, segments))


    def test_latvia_so_opener_does_not_prepend_fragment(self):
        segments = [
            {"start": 1559.0, "end": 1565.16, "text": "Life in Stokes."},
            {"start": 1565.16, "end": 1582.0, "text": "So almost all my clients are Latvian."},
            {"start": 1582.0, "end": 1622.74, "text": "Their portfolios need liquidity and diversification."},
        ]
        item = finalist(1565.16, 1622.74)
        review = {
            "start_ok": False,
            "end_ok": True,
            "needs_repair": True,
            "original_opening_syntactically_dependent": False,
            "original_opening_semantically_standalone": True,
            "proposed_repair_improves_opening": False,
            "proposed_repair_introduces_unrelated_or_fragmentary_material": True,
            "proposed_start_time": 1559.0,
            "proposed_end_time": 1622.74,
            "reason": "So is conversational; the earlier ASR fragment is worse.",
        }
        verified = _apply_final_boundary_review(item, review, segments)
        self.assertEqual(1565.16, verified["start_time"])
        self.assertEqual("original opening", verified["actual_opening_quote"])
        self.assertFalse(verified["final_boundary_repaired"])

    def test_bitcoin_and_opener_is_semantically_allowed(self):
        segments = [
            {"start": 1307.60, "end": 1325.0, "text": "And every single narrative I've seen with Bitcoin has failed."},
            {"start": 1325.0, "end": 1360.88, "text": "That is why technical indicators are not enough."},
        ]
        item = finalist(1307.60, 1360.88)
        review = {
            "start_ok": True,
            "end_ok": True,
            "needs_repair": False,
            "original_opening_syntactically_dependent": False,
            "original_opening_semantically_standalone": True,
            "proposed_repair_improves_opening": False,
            "proposed_repair_introduces_unrelated_or_fragmentary_material": False,
            "proposed_start_time": 1307.60,
            "proposed_end_time": 1360.88,
            "reason": "And is a harmless discourse marker here.",
        }
        verified = _apply_final_boundary_review(item, review, segments)
        self.assertEqual(1307.60, verified["start_time"])
        self.assertFalse(verified["final_boundary_repaired"])

    def test_unrelated_previous_fragment_cannot_be_accepted_as_repair(self):
        segments = [
            {"start": 5.0, "end": 10.0, "text": "Life in Stokes."},
            {"start": 10.0, "end": 25.0, "text": "The selected idea continues."},
            {"start": 25.0, "end": 40.0, "text": "It reaches a complete payoff."},
        ]
        review = {
            "start_ok": False,
            "end_ok": True,
            "needs_repair": True,
            "proposed_start_time": 5.0,
            "proposed_end_time": 40.0,
            **dependent_opening_repair_ok(),
            "proposed_repair_introduces_unrelated_or_fragmentary_material": True,
            "reason": "The previous segment is an unrelated ASR fragment.",
        }
        self.assertIsNone(_apply_final_boundary_review(finalist(10.0, 40.0), review, segments))

    def test_final_evidence_has_three_previous_segments_and_opening_options(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Earlier one."},
            {"start": 5.0, "end": 10.0, "text": "Earlier two."},
            {"start": 10.0, "end": 15.0, "text": "The reason is that"},
            {"start": 15.0, "end": 25.0, "text": "the candidate depends on it."},
            {"start": 25.0, "end": 40.0, "text": "The payoff follows."},
        ]
        evidence = _final_boundary_evidence(finalist(15.0, 40.0), segments)
        self.assertEqual(3, len(evidence["previous_segments"]))
        self.assertEqual([0.0, 5.0, 10.0], [
            option["proposed_start_time"]
            for option in evidence["candidate_start_options"]
        ])
        self.assertIn("the candidate depends on it.", [
            segment["text"]
            for segment in evidence["candidate_start_options"][-1]["opening_segments"]
        ])
if __name__ == "__main__":
    unittest.main()
