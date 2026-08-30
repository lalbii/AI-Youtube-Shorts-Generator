import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import main as cli
from scripts.render_selected import load_selected_highlights
from shorts_generator import pipeline


HIGHLIGHT = {
    "candidate_id": "candidate_001",
    "title": "Selected insight",
    "start_time": 10.0,
    "end_time": 40.0,
    "score": 88.0,
    "final_score": 88.0,
    "hook_sentence": "A useful hook",
}


def fake_result(render=False):
    rendered = {**HIGHLIGHT, "clip_url": "output/short_01.mp4"}
    return {
        "mode": "local",
        "source_video_url": "source.mp4",
        "transcript": {"segments": []},
        "candidates": [{"candidate_id": "candidate_001"}],
        "highlights": [dict(HIGHLIGHT)],
        "shorts": [rendered] if render else [dict(HIGHLIGHT)],
    }


class MainRenderFlagTests(unittest.TestCase):
    def invoke(self, extra_args):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_json = root / "selection.json"
            calls = []

            def generator(**kwargs):
                calls.append(kwargs)
                return fake_result(render=kwargs["render"])

            argv = [
                "main.py", "source.mp4", "--mode", "local",
                "--output-json", str(output_json), *extra_args,
            ]
            stdout = io.StringIO()
            previous_cwd = os.getcwd()
            try:
                os.chdir(root)
                with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    cli, "generate_shorts", side_effect=generator
                ), redirect_stdout(stdout):
                    exit_code = cli.main()
            finally:
                os.chdir(previous_cwd)
            consumed = load_selected_highlights(output_json)
            return (
                exit_code,
                calls,
                json.loads(output_json.read_text(encoding="utf-8")),
                stdout.getvalue(),
                list(root.rglob("*.mp4")),
                consumed,
            )

    def test_default_writes_json_without_rendering_or_fake_clip_path(self):
        exit_code, calls, result, output, mp4s, consumed = self.invoke([])
        self.assertEqual(0, exit_code)
        self.assertFalse(calls[0]["render"])
        self.assertTrue(result["highlights"])
        self.assertTrue(result["shorts"])
        self.assertEqual(result["highlights"], result["shorts"])
        self.assertEqual([], mp4s)
        self.assertNotIn("clip:", output)
        self.assertIn("Selection JSON written to", output)
        self.assertIn("Selected insight", output)

        self.assertEqual(result["highlights"], consumed)

    def test_render_flag_invokes_existing_render_path_and_prints_real_clip(self):
        exit_code, calls, result, output, _, _ = self.invoke(["--render"])
        self.assertEqual(0, exit_code)
        self.assertTrue(calls[0]["render"])
        self.assertEqual("output/short_01.mp4", result["shorts"][0]["clip_url"])
        self.assertIn("clip:   output/short_01.mp4", output)

    def test_render_specific_flags_do_not_enable_rendering(self):
        _, calls, _, output, _, _ = self.invoke(
            ["--aspect-ratio", "1:1", "--crop-mode", "face"]
        )
        self.assertFalse(calls[0]["render"])
        self.assertEqual("1:1", calls[0]["aspect_ratio"])
        self.assertEqual("face", calls[0]["crop_mode"])
        self.assertNotIn("clip:", output)

    def test_selection_content_is_identical_with_and_without_render(self):
        _, _, without_render, _, _, _ = self.invoke([])
        _, _, with_render, _, _, _ = self.invoke(["--render"])
        self.assertEqual(without_render["candidates"], with_render["candidates"])
        self.assertEqual(without_render["highlights"], with_render["highlights"])


class PipelineRenderControlTests(unittest.TestCase):
    def test_api_renderer_is_called_only_when_enabled(self):
        transcript = {"segments": [{"start": 0.0, "end": 1.0, "text": "speech"}]}
        selection = {"highlights": [dict(HIGHLIGHT)], "candidates": [{}]}
        rendered = [{**HIGHLIGHT, "clip_url": "rendered.mp4"}]
        with mock.patch.object(
            pipeline, "download_youtube", return_value="source.mp4"
        ), mock.patch.object(
            pipeline, "transcribe", return_value=transcript
        ), mock.patch.object(
            pipeline, "get_highlights", return_value=selection
        ), mock.patch(
            "shorts_generator.clipper.crop_highlights", return_value=rendered
        ) as renderer:
            selected_only = pipeline._run_api(
                "source", 1, "9:16", "720", "en", False
            )
            renderer.assert_not_called()
            with_render = pipeline._run_api(
                "source", 1, "9:16", "720", "en", True
            )
            renderer.assert_called_once_with(
                "source.mp4", [HIGHLIGHT], aspect_ratio="9:16"
            )

        self.assertEqual(selected_only["highlights"], selected_only["shorts"])
        self.assertEqual(rendered, with_render["shorts"])
        self.assertEqual(selected_only["highlights"], with_render["highlights"])


if __name__ == "__main__":
    unittest.main()
