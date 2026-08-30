import json
import unittest

from shorts_generator.highlights import (
    _apply_final_boundary_review,
    select_finalists_with_boundary_qa,
)


def candidate(name, score, start):
    return {
        "candidate_id": name,
        "title": name,
        "start_time": float(start),
        "end_time": float(start + 20),
        "actual_opening_quote": f"{name} opening.",
        "actual_closing_quote": f"{name} ending.",
        "opening_complete": True,
        "ending_complete": True,
        "standalone_context": True,
        "has_payoff": True,
        "publishable": True,
        "final_score": float(score),
        "score": float(score),
        "scores": {"actual_hook_strength": score / 10},
        "topic_key": f"topic {name}",
    }


def transcript_segments(candidates):
    return [
        {
            "start": item["start_time"],
            "end": item["end_time"],
            "text": f"{item['candidate_id']} complete standalone passage.",
        }
        for item in candidates
    ]


def critic(failed_ids, observed_pool=None):
    def fake_llm(prompt):
        evidence = json.loads(
            prompt.split("Finalists and local transcript evidence:\n", 1)[1]
        )
        if observed_pool is not None:
            observed_pool.extend(item["candidate_id"] for item in evidence)
        reviews = []
        for item in evidence:
            failed = item["candidate_id"] in failed_ids
            reviews.append({
                "candidate_id": item["candidate_id"],
                "start_ok": True,
                "end_ok": not failed,
                "needs_repair": failed,
                "proposed_start_time": item["proposed_start_time"],
                "proposed_end_time": 999.0 if failed else item["proposed_end_time"],
                "reason": "ending cannot be repaired" if failed else "clean",
            })
        return json.dumps({"reviews": reviews})
    return fake_llm


class FinalistBackfillTests(unittest.TestCase):
    def setUp(self):
        self.ranked = [
            candidate("A", 95, 0),
            candidate("B", 90, 30),
            candidate("C", 85, 60),
            candidate("D", 80, 90),
            candidate("E", 75, 120),
        ]
        self.segments = transcript_segments(self.ranked)

    def test_failed_third_finalist_is_backfilled_by_fourth(self):
        result = select_finalists_with_boundary_qa(
            self.ranked[:4], self.segments[:4], 3, critic({"C"})
        )
        self.assertEqual(["A", "B", "D"], [item["candidate_id"] for item in result])

    def test_top_three_pass_unchanged(self):
        result = select_finalists_with_boundary_qa(
            self.ranked, self.segments, 3, critic(set())
        )
        self.assertEqual(["A", "B", "C"], [item["candidate_id"] for item in result])

    def test_multiple_failures_backfill_in_ranking_order(self):
        result = select_finalists_with_boundary_qa(
            self.ranked, self.segments, 3, critic({"B", "C"})
        )
        self.assertEqual(["A", "D", "E"], [item["candidate_id"] for item in result])

    def test_not_enough_survivors_returns_fewer(self):
        result = select_finalists_with_boundary_qa(
            self.ranked[:4], self.segments[:4], 3, critic({"B", "C", "D"})
        )
        self.assertEqual(["A"], [item["candidate_id"] for item in result])

    def test_more_than_sixty_second_repair_remains_rejected(self):
        item = candidate("candidate_006", 82, 0)
        item["end_time"] = 60.0
        segments = [
            {"start": 0.0, "end": 30.0, "text": "The claim begins."},
            {"start": 30.0, "end": 60.0, "text": "It ends with a lot of people"},
            {"start": 60.0, "end": 65.0, "text": "and only then completes."},
        ]
        review = {
            "start_ok": True,
            "end_ok": False,
            "needs_repair": True,
            "proposed_start_time": 0.0,
            "proposed_end_time": 65.0,
            "repaired_opening_complete": True,
            "repaired_ending_complete": True,
            "repaired_standalone_context": True,
            "repaired_has_payoff": True,
            "repaired_publishable": True,
            "reason": "Completion exceeds sixty seconds.",
        }
        self.assertIsNone(_apply_final_boundary_review(item, review, segments))

    def test_backfill_does_not_recompute_scores(self):
        result = select_finalists_with_boundary_qa(
            self.ranked, self.segments, 3, critic({"C"})
        )
        self.assertEqual([95.0, 90.0, 80.0], [item["final_score"] for item in result])
        self.assertEqual(self.ranked[3]["scores"], result[2]["scores"])

    def test_final_critic_pool_is_bounded(self):
        ranked = [candidate(chr(65 + index), 100 - index, index * 30) for index in range(10)]
        observed = []
        result = select_finalists_with_boundary_qa(
            ranked, transcript_segments(ranked), 3, critic(set(), observed)
        )
        self.assertEqual(6, len(observed))
        self.assertEqual(["A", "B", "C"], [item["candidate_id"] for item in result])


if __name__ == "__main__":
    unittest.main()
