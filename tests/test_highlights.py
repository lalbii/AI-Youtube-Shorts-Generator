import unittest

from shorts_generator.highlights import get_highlights


class LongVideoChunkingTests(unittest.TestCase):
    def test_second_chunk_uses_relative_times_and_returns_global_times(self):
        transcript = {
            "duration": 1801.0,
            "segments": [
                {"start": 10.0, "end": 20.0, "text": "first chunk"},
                {"start": 1150.0, "end": 1160.0, "text": "second chunk hook"},
                {"start": 1790.0, "end": 1801.0, "text": "second chunk ending"},
            ],
        }
        prompts = []

        def fake_llm(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"content_type":"podcast","density":"medium"}'
            if len(prompts) == 2:
                return '{"highlights":[{"title":"first","start_time":10,"end_time":50,"score":80}]}'
            return '{"highlights":[{"title":"second","start_time":10,"end_time":60,"score":90}]}'

        result = get_highlights(transcript, num_clips=1, llm_fn=fake_llm)

        self.assertIn("[10.0s] second chunk hook", prompts[2])
        self.assertNotIn("[1150.0s]", prompts[2])
        second = next(h for h in result["highlights"] if h["title"] == "second")
        self.assertEqual(1150.0, second["start_time"])
        self.assertEqual(1200.0, second["end_time"])


if __name__ == "__main__":
    unittest.main()
