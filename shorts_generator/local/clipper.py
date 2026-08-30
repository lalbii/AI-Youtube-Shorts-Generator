"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio with a center, static-face,
     dynamic face crop, or a full-frame fit over a blurred background. The
     OpenCV/mp4v video used by crop modes is only an intermediate artifact;
     every final MP4 is explicitly encoded as social-media-ready H.264/AAC.
"""
import os
import statistics
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


CROP_MODES = ("center", "static-face", "face", "fit-blur")
STATIC_FACE_SAMPLE_COUNT = 9


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _validate_crop_mode(crop_mode: str) -> str:
    if crop_mode not in CROP_MODES:
        choices = ", ".join(CROP_MODES)
        raise ValueError(f"unknown crop mode {crop_mode!r}; choose one of: {choices}")
    return crop_mode


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _final_encode_command(
    silent_path: str,
    audio_path: str,
    out_path: str,
    aspect_ratio: str,
) -> List[str]:
    """Build the final H.264/AAC encode command for an OpenCV intermediate."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0?",
    ]
    if _is_9_16(aspect_ratio):
        # OpenCV has already made the correctly framed 9:16 crop. Scaling here
        # changes only its publish resolution; it does not repeat or alter the
        # face-aware crop.
        cmd.extend(["-vf", "scale=1080:1920:flags=lanczos"])
    cmd.extend(_publish_encode_options())
    cmd.extend([
        "-shortest",
        out_path,
    ])
    return cmd


def _is_9_16(aspect_ratio: str) -> bool:
    return abs(_ratio(aspect_ratio) - (9.0 / 16.0)) < 1e-9


def _publish_encode_options() -> List[str]:
    """Codec settings shared by every final local-render deliverable."""
    return [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
    ]


def _fit_blur_filter() -> str:
    """Build a 9:16 blurred-cover background plus uncropped fitted foreground."""
    return (
        "[0:v]split=2[background][foreground];"
        "[background]"
        "scale=1080:1920:force_original_aspect_ratio=increase:force_divisible_by=2,"
        "crop=1080:1920,"
        "gblur=sigma=30,"
        "eq=brightness=-0.08[blurred];"
        "[foreground]"
        "scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2"
        "[fitted];"
        "[blurred][fitted]overlay=(W-w)/2:(H-h)/2:shortest=1[video]"
    )


def _fit_blur_encode_command(in_path: str, out_path: str, aspect_ratio: str) -> List[str]:
    """Build the direct FFmpeg fit-blur render command for a cut clip."""
    if not _is_9_16(aspect_ratio):
        raise ValueError("fit-blur currently requires a 9:16 output aspect ratio")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-filter_complex", _fit_blur_filter(),
        "-map", "[video]", "-map", "0:a:0?",
    ]
    cmd.extend(_publish_encode_options())
    cmd.extend(["-shortest", out_path])
    return cmd


def _primary_face_center(frame, face_cascade, cv2) -> Optional[Tuple[int, int]]:
    """Return the center of the largest detected face in one frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return x + w // 2, y + h // 2


def _static_face_center_x(
    cap,
    face_cascade,
    src_w: int,
    cv2,
    sample_count: int = STATIC_FACE_SAMPLE_COUNT,
) -> int:
    """Choose one median face X from frames sampled across the cut clip."""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0 or sample_count <= 0:
        return src_w // 2

    actual_count = min(sample_count, frame_count)
    if actual_count == 1:
        frame_indexes = [0]
    else:
        frame_indexes = sorted({
            round(index * (frame_count - 1) / (actual_count - 1))
            for index in range(actual_count)
        })

    face_centers_x: List[int] = []
    try:
        for frame_index in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            center = _primary_face_center(frame, face_cascade, cv2)
            if center is not None:
                face_centers_x.append(center[0])
    finally:
        # Static sampling must not change the sequential render's start frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not face_centers_x:
        return src_w // 2
    return int(statistics.median(face_centers_x))


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    crop_mode: str = "face",
) -> str:
    """Lay out the cut clip using the requested local crop mode."""
    crop_mode = _validate_crop_mode(crop_mode)
    if crop_mode == "fit-blur":
        # This path intentionally avoids OpenCV and Haar face detection. FFmpeg
        # fits the complete source over a blurred cover background in one pass.
        cmd = _fit_blur_encode_command(in_path, out_path, aspect_ratio)
        subprocess.run(cmd, check=True)
        return out_path

    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    fixed_center: Optional[Tuple[int, int]] = None
    if crop_mode == "center":
        fixed_center = (src_w // 2, src_h // 2)
    elif crop_mode == "static-face":
        fixed_center = (
            _static_face_center_x(cap, face_cascade, src_w, cv2),
            src_h // 2,
        )

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if crop_mode == "face":
            center = _primary_face_center(frame, face_cascade, cv2)
            if center is not None:
                cx, cy = center
                if last_center is None:
                    last_center = (cx, cy)
                else:
                    lx, ly = last_center
                    last_center = (
                        int(lx + (cx - lx) * smoothing),
                        int(ly + (cy - ly) * smoothing),
                    )
            if last_center is None:
                last_center = (src_w // 2, src_h // 2)
            cx, cy = last_center
        else:
            # Center/static-face select this once before rendering. It remains
            # fixed for every frame, eliminating detector-induced jitter.
            assert fixed_center is not None
            cx, cy = fixed_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()

    # The OpenCV mp4v file is deliberately temporary. Re-encode it instead of
    # copying that codec into the deliverable, and mux the cut clip's audio.
    cmd = _final_encode_command(silent_path, in_path, out_path, aspect_ratio)
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    crop_mode: str = "face",
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, crop_mode=crop_mode)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    crop_mode: str = "face",
) -> List[Dict]:
    """Render highlights in order; dynamic face tracking remains the default."""
    crop_mode = _validate_crop_mode(crop_mode)
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                crop_mode=crop_mode,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
