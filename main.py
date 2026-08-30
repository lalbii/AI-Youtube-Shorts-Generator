"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts
CROP_MODES = ("center", "static-face", "face", "fit-blur")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url", help="YouTube URL, file:// URL, or local file path")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI) or local (remote URL, file://, or local path + faster-whisper + LLM provider + ffmpeg).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many clips to select (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument(
        "--crop-mode",
        choices=CROP_MODES,
        default="face",
        help="Local render layout (default: face; ignored in API mode)",
    )
    parser.add_argument("--output-json", default="clip_final.json", help="Write the full selection JSON to this path (default: clip_final.json)")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render selected clips after writing the selection JSON. Disabled by default.",
    )
    args = parser.parse_args()

    try:
        result = generate_shorts(
            youtube_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
            crop_mode=args.crop_mode,
            render=args.render,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', args.mode)}")
    print(f"Source video:  {result['source_video_url']}")
    print(
        f"Highlights:    {len(result.get('candidates', result['highlights']))} candidates "
        f"→ {len(result['highlights'])} publishable"
    )
    print("=" * 72)
    selected = result["shorts"] if args.render else result["highlights"]
    for i, s in enumerate(selected, 1):
        print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
        print(f"     title:  {s.get('title')}")
        print(f"     hook:   {s.get('hook_sentence')}")
        if args.render:
            if s.get("clip_url"):
                print(f"     clip:   {s['clip_url']}")
            else:
                print(f"     clip:   FAILED ({s.get('error')})")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSelection JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
