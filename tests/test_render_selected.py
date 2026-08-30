from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.render_selected import (
    SelectionError,
    _parser,
    load_selected_highlights,
    render_selected,
)


class RenderSelectedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_json(self, value: object) -> Path:
        path = self.root / "selection.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_load_preserves_shorts_order_and_ignores_candidates(self) -> None:
        selection = self.write_json(
            {
                "highlights": [
                    {"title": "candidate only", "start_time": 1, "end_time": 2}
                ],
                "shorts": [
                    {
                        "title": "second chronologically",
                        "start_time": 20,
                        "end_time": 25,
                        "clip_url": "old.mp4",
                    },
                    {
                        "title": "first chronologically",
                        "start_time": 5.5,
                        "end_time": 9,
                    },
                ],
            }
        )

        highlights = load_selected_highlights(selection)

        self.assertEqual(
            [item["title"] for item in highlights],
            ["second chronologically", "first chronologically"],
        )
        self.assertEqual(highlights[0]["start_time"], 20.0)
        self.assertNotIn("clip_url", highlights[0])

    def test_rejects_malformed_selections(self) -> None:
        cases = [
            ({}, "non-empty 'shorts' array"),
            ({"shorts": []}, "non-empty 'shorts' array"),
            (
                {"shorts": [{"start_time": 3, "end_time": 2}]},
                "greater than start_time",
            ),
            (
                {"shorts": [{"start_time": "3", "end_time": 4}]},
                "must be a number",
            ),
        ]

        for document, message in cases:
            with self.subTest(document=document):
                selection = self.write_json(document)
                with self.assertRaisesRegex(SelectionError, message):
                    load_selected_highlights(selection)

    def test_rejects_invalid_json_with_location(self) -> None:
        selection = self.root / "selection.json"
        selection.write_text('{"shorts": [}', encoding="utf-8")

        with self.assertRaisesRegex(SelectionError, r"line 1, column \d+"):
            load_selected_highlights(selection)

    def test_cli_accepts_fit_blur_crop_mode(self) -> None:
        arguments = _parser().parse_args(
            [
                "--source", "source.mp4",
                "--selection-json", "selection.json",
                "--output-dir", "output",
                "--crop-mode", "fit-blur",
            ]
        )

        self.assertEqual(arguments.crop_mode, "fit-blur")

    def test_rerender_imports_only_local_clipper_and_keeps_order(self) -> None:
        source = self.root / "source.mp4"
        source.touch()
        selection = self.write_json(
            {
                "shorts": [
                    {"title": "B", "start_time": 30, "end_time": 35},
                    {"title": "A", "start_time": 10, "end_time": 15},
                ]
            }
        )
        calls: list[tuple[str, list[str], str, str, str]] = []

        def local_renderer(source_path, highlights, aspect_ratio, out_dir, *, crop_mode):
            calls.append(
                (
                    source_path,
                    [highlight["title"] for highlight in highlights],
                    aspect_ratio,
                    out_dir,
                    crop_mode,
                )
            )
            return [
                {
                    **highlight,
                    "clip_url": str(Path(out_dir) / f"short_{index:02d}.mp4"),
                }
                for index, highlight in enumerate(highlights, 1)
            ]

        package = types.ModuleType("shorts_generator")
        package.__path__ = []
        local_package = types.ModuleType("shorts_generator.local")
        local_package.__path__ = []
        clipper = types.ModuleType("shorts_generator.local.clipper")
        clipper.crop_highlights_local = local_renderer

        # These sentinels represent the expensive/remote pipeline stages. The
        # default rerender path must never import or invoke any of them.
        transcriber = types.ModuleType("shorts_generator.local.transcriber")
        transcriber.transcribe_local = Mock(side_effect=AssertionError("Whisper called"))
        highlights = types.ModuleType("shorts_generator.highlights")
        highlights.find_highlights = Mock(side_effect=AssertionError("selection called"))
        pipeline = types.ModuleType("shorts_generator.pipeline")
        pipeline.generate_shorts = Mock(side_effect=AssertionError("pipeline called"))

        with patch.dict(
            sys.modules,
            {
                "shorts_generator": package,
                "shorts_generator.local": local_package,
                "shorts_generator.local.clipper": clipper,
                "shorts_generator.local.transcriber": transcriber,
                "shorts_generator.highlights": highlights,
                "shorts_generator.pipeline": pipeline,
            },
        ):
            results = render_selected(source, selection, self.root / "out")

        transcriber.transcribe_local.assert_not_called()
        highlights.find_highlights.assert_not_called()
        pipeline.generate_shorts.assert_not_called()

        self.assertEqual(
            calls,
            [(str(source), ["B", "A"], "9:16", str(self.root / "out"), "static-face")],
        )
        self.assertEqual(
            [Path(result["clip_url"]).name for result in results],
            ["short_01.mp4", "short_02.mp4"],
        )


if __name__ == "__main__":
    unittest.main()
