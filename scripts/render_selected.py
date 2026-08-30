#!/usr/bin/env python3
"""Rerender previously selected highlights from a local source video.

This entry point deliberately imports only the existing local clipper.  It does
not run the normal pipeline, transcription, content classification, highlight
selection, or any remote service.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


# ``python scripts/render_selected.py`` puts scripts/, rather than the project
# root, on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SelectionError(ValueError):
    """The selector JSON cannot safely be used for rerendering."""


Renderer = Callable[..., list[dict[str, Any]]]
CROP_MODES = ("center", "static-face", "face", "fit-blur")


def _timestamp(value: Any, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"shorts[{index}].{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionError(f"shorts[{index}].{field} must be finite")
    return result


def load_selected_highlights(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the selected ``shorts`` list, preserving its order."""
    json_path = Path(path)
    if not json_path.is_file():
        raise SelectionError(f"selection JSON does not exist: {json_path}")

    try:
        with json_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SelectionError(
            f"malformed selection JSON {json_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except UnicodeError as exc:
        raise SelectionError(f"selection JSON is not valid UTF-8: {json_path}: {exc}") from exc
    except OSError as exc:
        raise SelectionError(f"could not read selection JSON {json_path}: {exc}") from exc

    if not isinstance(document, dict):
        raise SelectionError("selection JSON root must be an object")

    selected = document.get("shorts")
    if not isinstance(selected, list) or not selected:
        raise SelectionError(
            "selection JSON must contain a non-empty 'shorts' array of selected highlights"
        )

    highlights: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise SelectionError(f"shorts[{index}] must be an object")
        if "start_time" not in item or "end_time" not in item:
            raise SelectionError(
                f"shorts[{index}] must contain 'start_time' and 'end_time'"
            )

        start = _timestamp(item["start_time"], "start_time", index)
        end = _timestamp(item["end_time"], "end_time", index)
        if start < 0:
            raise SelectionError(f"shorts[{index}].start_time must be non-negative")
        if end <= start:
            raise SelectionError(f"shorts[{index}].end_time must be greater than start_time")

        highlight = dict(item)
        highlight["start_time"] = start
        highlight["end_time"] = end
        # These describe an earlier render, not the selection itself.
        highlight.pop("clip_url", None)
        highlight.pop("error", None)
        highlights.append(highlight)

    return highlights


def _validate_aspect_ratio(value: str) -> str:
    try:
        width, height = (float(part) for part in value.split(":"))
    except (TypeError, ValueError):
        raise SelectionError("aspect ratio must be WIDTH:HEIGHT, for example 9:16") from None
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise SelectionError("aspect ratio dimensions must be positive finite numbers")
    return value


def _validate_crop_mode(value: str) -> str:
    if value not in CROP_MODES:
        choices = ", ".join(CROP_MODES)
        raise SelectionError(f"crop mode must be one of: {choices}")
    return value


def _get_renderer() -> Renderer:
    # Import the leaf renderer only. Importing generate_shorts/pipeline here
    # would make accidental transcription or selection much easier.
    try:
        from shorts_generator.local.clipper import crop_highlights_local
    except ImportError as exc:
        raise RuntimeError(
            "could not import the existing local renderer "
            "(shorts_generator.local.clipper); run this script from the repository "
            "with local rendering dependencies installed"
        ) from exc
    return crop_highlights_local


def render_selected(
    source: str | Path,
    selection_json: str | Path,
    output_dir: str | Path,
    aspect_ratio: str = "9:16",
    crop_mode: str = "static-face",
    *,
    renderer: Renderer | None = None,
) -> list[dict[str, Any]]:
    """Render selected highlights only, using the repository's local clipper."""
    source_path = Path(source)
    if not source_path.is_file():
        raise SelectionError(f"source video does not exist: {source_path}")

    highlights = load_selected_highlights(selection_json)
    ratio = _validate_aspect_ratio(aspect_ratio)
    selected_crop_mode = _validate_crop_mode(crop_mode)
    destination = Path(output_dir)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"could not create output directory {destination}: {exc}") from exc

    render = renderer or _get_renderer()
    results = render(
        str(source_path),
        highlights,
        ratio,
        str(destination),
        crop_mode=selected_crop_mode,
    )
    if not isinstance(results, list) or len(results) != len(highlights):
        raise RuntimeError(
            "local renderer returned an unexpected number of results "
            f"(expected {len(highlights)})"
        )

    failures: list[str] = []
    for index, (highlight, result) in enumerate(zip(highlights, results), 1):
        title = str(highlight.get("title") or f"Untitled highlight {index}")
        start = highlight["start_time"]
        end = highlight["end_time"]
        duration = end - start
        output_path = result.get("clip_url") if isinstance(result, dict) else None

        print(f"#{index} {title}")
        print(f"  start/end: {start:.3f}s -> {end:.3f}s")
        print(f"  duration:  {duration:.3f}s")
        print(f"  output:    {output_path or '(render failed)'}")

        if not output_path:
            detail = result.get("error", "unknown render error") if isinstance(result, dict) else "invalid renderer result"
            failures.append(f"#{index} {title}: {detail}")

    if failures:
        raise RuntimeError("one or more clips failed to render:\n  " + "\n  ".join(failures))
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerender already-selected highlights locally without transcription or APIs."
    )
    parser.add_argument("--source", required=True, help="High-quality local source video")
    parser.add_argument("--selection-json", required=True, help="Existing selector result JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered MP4 clips")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument(
        "--crop-mode",
        choices=CROP_MODES,
        default="static-face",
        help=(
            "Layout: fixed center, fixed median face, dynamic face, or full-frame "
            "fit over blur (default: static-face)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        render_selected(
            source=args.source,
            selection_json=args.selection_json,
            output_dir=args.output_dir,
            aspect_ratio=args.aspect_ratio,
            crop_mode=args.crop_mode,
        )
    except (SelectionError, RuntimeError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
