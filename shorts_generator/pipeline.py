"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.
"""
from typing import Dict, List, Optional

from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    crop_mode: str,
    render: bool,
) -> Dict:
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    source_path = download_youtube_local(youtube_url, fmt=download_format)

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    candidates: List[Dict] = highlights_result.get("candidates", [])
    if not all_highlights:
        raise RuntimeError("Highlight judge found no publishable clips.")

    finalists = sorted(
        all_highlights,
        key=lambda h: float(h.get("final_score", h.get("score", 0))),
        reverse=True,
    )[:num_clips]
    shorts: List[Dict] = list(finalists)
    if render:
        from .local.clipper import crop_highlights_local

        print(
            f"[pipeline/local] rendering {len(finalists)} publishable clips "
            f"from {len(candidates)} judged candidates",
            flush=True,
        )
        shorts = crop_highlights_local(
            source_path,
            finalists,
            aspect_ratio=aspect_ratio,
            crop_mode=crop_mode,
        )

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "candidates": candidates,
        "highlights": finalists,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    render: bool,
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    candidates: List[Dict] = highlights_result.get("candidates", [])
    if not all_highlights:
        raise RuntimeError("Highlight judge found no publishable clips.")

    finalists = sorted(
        all_highlights,
        key=lambda h: float(h.get("final_score", h.get("score", 0))),
        reverse=True,
    )[:num_clips]
    shorts: List[Dict] = list(finalists)
    if render:
        from .clipper import crop_highlights

        print(
            f"[pipeline] rendering {len(finalists)} publishable clips "
            f"from {len(candidates)} judged candidates",
            flush=True,
        )
        shorts = crop_highlights(source_url, finalists, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "candidates": candidates,
        "highlights": finalists,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    crop_mode: str = "face",
    render: bool = False,
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many clips to select.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg).
        crop_mode: local renderer layout: "center", "static-face", "face", or
            "fit-blur". Ignored in API mode.
        render: render selected clips after selection. Disabled by default.

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "candidates": [...],       # all exact-range judgments
          "highlights": [...],       # publishable finalists, deterministically ranked
          "shorts": [...],           # selected finalists; rendered metadata when enabled
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(
            youtube_url,
            num_clips,
            aspect_ratio,
            download_format,
            language,
            crop_mode,
            render,
        )
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language, render)
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
