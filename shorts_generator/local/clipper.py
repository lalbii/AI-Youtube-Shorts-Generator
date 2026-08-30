"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio with a center, static-face,
     dynamic face crop, smart scene layout, or a full-frame fit over a blurred background. The
     OpenCV/mp4v video used by crop modes is only an intermediate artifact;
     every final MP4 is explicitly encoded as social-media-ready H.264/AAC.
"""
import math
import os
import statistics
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


CROP_MODES = ("center", "static-face", "face", "fit-blur", "smart-layout")
STATIC_FACE_SAMPLE_COUNT = 9
SMART_LAYOUT_WINDOW_SECONDS_ENV = "SMART_LAYOUT_WINDOW_SECONDS"
SMART_LAYOUT_WINDOW_SECONDS_DEFAULT = 3.0
SMART_LAYOUT_WINDOW_SECONDS_MIN = 0.5
SMART_LAYOUT_WINDOW_SECONDS_MAX = 30.0
SMART_SAMPLES_PER_WINDOW = 7
SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS_ENV = "SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS"
SMART_LAYOUT_ROLLING_WINDOW_SECONDS_ENV = "SMART_LAYOUT_ROLLING_WINDOW_SECONDS"
SMART_LAYOUT_SCENE_CUT_THRESHOLD_ENV = "SMART_LAYOUT_SCENE_CUT_THRESHOLD"
SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS_DEFAULT = 0.25
SMART_LAYOUT_ROLLING_WINDOW_SECONDS_DEFAULT = 1.25
SMART_LAYOUT_SCENE_CUT_THRESHOLD_DEFAULT = 0.22
SMART_LAYOUT_TRANSITION_CONFIRM_SAMPLES = 3
SMART_LAYOUT_SCENE_CUT_CONFIRM_SAMPLES = 2
SMART_LAYOUT_AMBIGUOUS_FALLBACK_SAMPLES = 6
SMART_LAYOUT_MIN_BLOCK_SECONDS = 1.0
SMART_LAYOUT_SCENE_SIGNATURE_SIZE = (160, 90)
SMART_MIN_HOLD_SECONDS = 2.5
SMART_RELIABILITY_RATIO = 0.6
SMART_SINGLE_RELIABILITY_RATIO = 0.5
SMART_YUNET_CONFIDENCE = 0.75
SMART_YUNET_NMS_THRESHOLD = 0.3
SMART_YUNET_TOP_K = 5000
SMART_MIN_FACE_AREA_RATIO = 0.001
SMART_DUPLICATE_IOU_THRESHOLD = 0.35
SMART_DUPLICATE_CENTER_RATIO = 0.35
SMART_MIN_PERSON_X_SEPARATION = 0.20
SMART_MAX_CLUSTER_X_SPREAD = 0.15
SMART_AMBIGUOUS_HOLD_WINDOWS = 1
SMART_YUNET_MODEL_ENV = "SMART_LAYOUT_YUNET_MODEL"
SMART_YUNET_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


FaceBox = Tuple[int, int, int, int]
CropBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class LayoutBlock:
    """A fixed composition applied for a contiguous section of a clip."""

    start: float
    end: float
    mode: str
    faces: Tuple[FaceBox, ...] = ()
    raw_mode: Optional[str] = None
    people: int = 0
    stabilization: str = ""
    cut_search_start: Optional[float] = None
    coarse_cut: Optional[float] = None
    refined_cut: Optional[float] = None
    cut_frame: Optional[int] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


@dataclass(frozen=True)
class FaceDecision:
    """Raw face-composition evidence for one analysis window."""

    mode: str
    faces: Tuple[FaceBox, ...]
    people: int


@dataclass(frozen=True)
class TimedFaceObservation:
    """One sampled frame's deduplicated face evidence."""

    timestamp: float
    faces: Tuple[FaceBox, ...]
    scene_cut: bool = False


@dataclass(frozen=True)
class SmartLayoutTemporalConfig:
    sample_interval: float
    rolling_window: float
    scene_cut_threshold: float


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


def _smart_layout_window_seconds() -> float:
    """Read and validate the smart-layout analysis-window duration."""
    raw_value = os.environ.get(
        SMART_LAYOUT_WINDOW_SECONDS_ENV,
        str(SMART_LAYOUT_WINDOW_SECONDS_DEFAULT),
    ).strip()
    try:
        value = float(raw_value)
    except ValueError:
        raise ValueError(
            f"{SMART_LAYOUT_WINDOW_SECONDS_ENV} must be a number between "
            f"{SMART_LAYOUT_WINDOW_SECONDS_MIN} and {SMART_LAYOUT_WINDOW_SECONDS_MAX} seconds"
        ) from None
    if (
        not math.isfinite(value)
        or value < SMART_LAYOUT_WINDOW_SECONDS_MIN
        or value > SMART_LAYOUT_WINDOW_SECONDS_MAX
    ):
        raise ValueError(
            f"{SMART_LAYOUT_WINDOW_SECONDS_ENV} must be between "
            f"{SMART_LAYOUT_WINDOW_SECONDS_MIN} and {SMART_LAYOUT_WINDOW_SECONDS_MAX} seconds"
        )
    return value


def _configured_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        raise ValueError(
            f"{name} must be a number between {minimum} and {maximum}"
        ) from None
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _smart_layout_temporal_config() -> SmartLayoutTemporalConfig:
    """Read the rolling analyzer's small, validated temporal configuration."""
    sample_interval = _configured_float(
        SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS_ENV,
        SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS_DEFAULT,
        0.1,
        2.0,
    )
    rolling_window = _configured_float(
        SMART_LAYOUT_ROLLING_WINDOW_SECONDS_ENV,
        SMART_LAYOUT_ROLLING_WINDOW_SECONDS_DEFAULT,
        0.5,
        5.0,
    )
    scene_cut_threshold = _configured_float(
        SMART_LAYOUT_SCENE_CUT_THRESHOLD_ENV,
        SMART_LAYOUT_SCENE_CUT_THRESHOLD_DEFAULT,
        0.05,
        0.95,
    )
    if rolling_window < sample_interval * 2:
        raise ValueError(
            f"{SMART_LAYOUT_ROLLING_WINDOW_SECONDS_ENV} must cover at least two "
            f"{SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS_ENV} intervals"
        )
    return SmartLayoutTemporalConfig(
        sample_interval,
        rolling_window,
        scene_cut_threshold,
    )


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


def _fit_blur_filter_segment(
    source: str = "0:v",
    output: str = "video",
    suffix: str = "",
) -> str:
    """Build the shared fit-blur graph for either a full clip or smart block."""
    return (
        f"[{source}]split=2[background{suffix}][foreground{suffix}];"
        f"[background{suffix}]"
        "scale=1080:1920:force_original_aspect_ratio=increase:force_divisible_by=2,"
        "crop=1080:1920,"
        "gblur=sigma=30,"
        f"eq=brightness=-0.08[blurred{suffix}];"
        f"[foreground{suffix}]"
        "scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2"
        f"[fitted{suffix}];"
        f"[blurred{suffix}][fitted{suffix}]"
        f"overlay=(W-w)/2:(H-h)/2:shortest=1[{output}]"
    )


def _fit_blur_filter() -> str:
    """Build a 9:16 blurred-cover background plus uncropped fitted foreground."""
    return _fit_blur_filter_segment()


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


def _create_smart_face_detector(cv2, model_path: Optional[str] = None):
    """Initialize the smart-layout-only YuNet detector from a local model."""
    configured_path = model_path or os.environ.get(SMART_YUNET_MODEL_ENV)
    path = Path(configured_path).expanduser() if configured_path else SMART_YUNET_MODEL_PATH
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError("OpenCV FaceDetectorYN/YuNet is unavailable")
    if not path.is_file():
        raise RuntimeError(
            f"YuNet model not found at {path}; set {SMART_YUNET_MODEL_ENV} to a local ONNX model"
        )
    try:
        return cv2.FaceDetectorYN_create(
            str(path),
            "",
            (320, 320),
            SMART_YUNET_CONFIDENCE,
            SMART_YUNET_NMS_THRESHOLD,
            SMART_YUNET_TOP_K,
        )
    except Exception as exc:
        raise RuntimeError(f"YuNet initialization failed for {path}: {exc}") from exc


def _face_iou(first: FaceBox, second: FaceBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / max(1, union)


def _duplicate_faces(first: FaceBox, second: FaceBox) -> bool:
    if _face_iou(first, second) >= SMART_DUPLICATE_IOU_THRESHOLD:
        return True
    first_center = (first[0] + first[2] / 2, first[1] + first[3] / 2)
    second_center = (second[0] + second[2] / 2, second[1] + second[3] / 2)
    distance = (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
    reference_size = max(first[2], first[3], second[2], second[3])
    return distance <= SMART_DUPLICATE_CENTER_RATIO * reference_size


def _deduplicate_faces(faces: List[FaceBox]) -> List[FaceBox]:
    """Suppress overlapping/near-identical detections before counting people."""
    kept: List[FaceBox] = []
    for face in sorted(faces, key=lambda item: item[2] * item[3], reverse=True):
        if not any(_duplicate_faces(face, existing) for existing in kept):
            kept.append(face)
    return sorted(kept, key=lambda face: face[0] + face[2] / 2)


def _distinct_people(first: FaceBox, second: FaceBox, frame_width: int) -> bool:
    first_x = first[0] + first[2] / 2
    second_x = second[0] + second[2] / 2
    separation = abs(second_x - first_x) / max(1, frame_width)
    return (
        not _duplicate_faces(first, second)
        and separation >= SMART_MIN_PERSON_X_SEPARATION
    )


def _detect_face_boxes(frame, face_detector, cv2) -> List[FaceBox]:
    """Return significant YuNet face boxes in stable left-to-right order."""
    height, width = int(frame.shape[0]), int(frame.shape[1])
    face_detector.setInputSize((width, height))
    _result, detected = face_detector.detect(frame)
    if detected is None:
        return []
    frame_area = max(1, int(frame.shape[0]) * int(frame.shape[1]))
    faces: List[FaceBox] = []
    for face in detected:
        raw_width = int(round(face[2]))
        raw_height = int(round(face[3]))
        if raw_width * raw_height < frame_area * SMART_MIN_FACE_AREA_RATIO:
            continue
        x = max(0, int(round(face[0])))
        y = max(0, int(round(face[1])))
        face_width = max(0, min(raw_width, width - x))
        face_height = max(0, min(raw_height, height - y))
        if face_width and face_height:
            faces.append((x, y, face_width, face_height))
    return _deduplicate_faces(faces)


def _median_face(faces: List[FaceBox]) -> FaceBox:
    """Combine repeated detections without allowing one outlier to steer a crop."""
    return tuple(int(statistics.median(values)) for values in zip(*faces))  # type: ignore[return-value]


def _cluster_is_consistent(faces: List[FaceBox], frame_width: int) -> bool:
    if not faces:
        return False
    centers = [face[0] + face[2] / 2 for face in faces]
    median_center = statistics.median(centers)
    return (
        sum(
            abs(center - median_center) / max(1, frame_width)
            <= SMART_MAX_CLUSTER_X_SPREAD
            for center in centers
        )
        >= max(2, int(len(centers) * SMART_RELIABILITY_RATIO + 0.999999))
    )


def _normalize_sample_faces(
    raw_faces: List[FaceBox],
    frame_width: int,
) -> List[FaceBox]:
    """Apply the existing duplicate/distinct-person rules to one observation."""
    faces = _deduplicate_faces(raw_faces)
    if len(faces) == 2 and not _distinct_people(faces[0], faces[1], frame_width):
        faces = [max(faces, key=lambda face: face[2] * face[3])]
    return faces


def _decide_face_samples(
    samples: List[List[FaceBox]],
    frame_width: int,
    reliability_ratio: float = SMART_RELIABILITY_RATIO,
) -> FaceDecision:
    """Derive persistent distinct people from all samples in one window."""
    if not samples:
        return FaceDecision("no-face", (), 0)

    normalized: List[List[FaceBox]] = []
    for raw_faces in samples:
        normalized.append(_normalize_sample_faces(raw_faces, frame_width))

    required = max(2, int(len(samples) * reliability_ratio + 0.999999))
    counts = [len(faces) for faces in normalized]
    if sum(count >= 3 for count in counts) >= required:
        return FaceDecision("crowded", (), 3)

    two_face_samples = [faces for faces in normalized if len(faces) == 2]
    if len(two_face_samples) >= required:
        left_faces = [faces[0] for faces in two_face_samples]
        right_faces = [faces[1] for faces in two_face_samples]
        left = _median_face(left_faces)
        right = _median_face(right_faces)
        if (
            _cluster_is_consistent(left_faces, frame_width)
            and _cluster_is_consistent(right_faces, frame_width)
            and _distinct_people(left, right, frame_width)
        ):
            return FaceDecision("split", (left, right), 2)

    one_face_samples = [faces[0] for faces in normalized if len(faces) == 1]
    single_required = max(
        2,
        int(len(samples) * SMART_SINGLE_RELIABILITY_RATIO + 0.999999),
    )
    if (
        len(one_face_samples) >= single_required
        and _cluster_is_consistent(one_face_samples, frame_width)
    ):
        return FaceDecision("single", (_median_face(one_face_samples),), 1)

    if two_face_samples:
        left = _median_face([faces[0] for faces in two_face_samples])
        right = _median_face([faces[1] for faces in two_face_samples])
        return FaceDecision("ambiguous", (left, right), 2)
    if one_face_samples:
        return FaceDecision("ambiguous", (_median_face(one_face_samples),), 1)
    return FaceDecision("no-face", (), 0)


def _classify_face_samples(
    samples: List[List[FaceBox]],
    reliability_ratio: float = SMART_RELIABILITY_RATIO,
    frame_width: int = 1000,
) -> Tuple[str, Tuple[FaceBox, ...]]:
    """Compatibility wrapper returning only renderable layout names."""
    decision = _decide_face_samples(samples, frame_width, reliability_ratio)
    mode = decision.mode if decision.mode in ("single", "split") else "fit-blur"
    faces = decision.faces if mode != "fit-blur" else ()
    return mode, faces


def _stabilize_layout_blocks(
    raw_blocks: List[LayoutBlock],
    ambiguous_hold_windows: int = SMART_AMBIGUOUS_HOLD_WINDOWS,
) -> List[LayoutBlock]:
    """Carry one weak window; strong single/split/crowded evidence switches immediately."""
    stabilized: List[LayoutBlock] = []
    weak_run = 0
    for index, block in enumerate(raw_blocks):
        raw_mode = block.mode
        if raw_mode in ("single", "split"):
            stabilized.append(
                LayoutBlock(
                    block.start,
                    block.end,
                    raw_mode,
                    block.faces,
                    raw_mode=raw_mode,
                    people=block.people,
                )
            )
            weak_run = 0
            continue
        if raw_mode == "crowded":
            stabilized.append(
                LayoutBlock(
                    block.start,
                    block.end,
                    "fit-blur",
                    (),
                    raw_mode=raw_mode,
                    people=block.people,
                )
            )
            weak_run = 0
            continue

        previous = stabilized[-1] if stabilized else None
        previous_stable = previous is not None and previous.mode in ("single", "split")
        next_raw = raw_blocks[index + 1].mode if index + 1 < len(raw_blocks) else None
        bridge = (
            previous_stable
            and weak_run < ambiguous_hold_windows
            and next_raw == previous.mode
        )
        carry = previous_stable and weak_run < ambiguous_hold_windows
        if bridge or carry:
            assert previous is not None
            expected_faces = 1 if previous.mode == "single" else 2
            faces = block.faces if len(block.faces) == expected_faces else previous.faces
            stabilized.append(
                LayoutBlock(
                    block.start,
                    block.end,
                    previous.mode,
                    faces,
                    raw_mode=raw_mode,
                    people=block.people,
                    stabilization="carry-forward",
                )
            )
        else:
            stabilized.append(
                LayoutBlock(
                    block.start,
                    block.end,
                    "fit-blur",
                    (),
                    raw_mode=raw_mode,
                    people=block.people,
                )
            )
        weak_run += 1
    return stabilized


def _merge_layout_blocks(
    blocks: List[LayoutBlock],
    min_hold: float = SMART_MIN_HOLD_SECONDS,
) -> List[LayoutBlock]:
    """Merge identical neighbors and absorb a too-short trailing decision."""
    def compatible_faces(left: Tuple[FaceBox, ...], right: Tuple[FaceBox, ...]) -> bool:
        if len(left) != len(right):
            return False
        for first, second in zip(left, right):
            first_x = first[0] + first[2] / 2
            first_y = first[1] + first[3] / 2
            second_x = second[0] + second[2] / 2
            second_y = second[1] + second[3] / 2
            tolerance = 2.5 * max(first[2], first[3], second[2], second[3])
            if ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5 > tolerance:
                return False
        return True

    merged: List[LayoutBlock] = []
    for block in blocks:
        if (
            merged
            and merged[-1].mode == block.mode
            and compatible_faces(merged[-1].faces, block.faces)
        ):
            previous = merged[-1]
            if previous.faces and block.faces:
                face_count = min(len(previous.faces), len(block.faces))
                faces = tuple(
                    _median_face([previous.faces[index], block.faces[index]])
                    for index in range(face_count)
                )
            else:
                faces = previous.faces or block.faces
            merged[-1] = LayoutBlock(previous.start, block.end, block.mode, faces)
        else:
            merged.append(block)

    if len(merged) > 1 and merged[-1].end - merged[-1].start < min_hold:
        tail = merged.pop()
        previous = merged[-1]
        merged[-1] = LayoutBlock(previous.start, tail.end, previous.mode, previous.faces)
    return merged


def _scene_signature(frame, cv2):
    """Build a small blurred grayscale signature for conservative cut detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, SMART_LAYOUT_SCENE_SIGNATURE_SIZE)
    return cv2.GaussianBlur(resized, (5, 5), 0)


def _scene_change_score(previous_signature, current_signature, cv2) -> float:
    return float(cv2.absdiff(previous_signature, current_signature).mean()) / 255.0


def _robust_candidate_faces(
    decisions: List[FaceDecision],
    mode: str,
) -> Tuple[FaceBox, ...]:
    expected = 1 if mode == "single" else 2 if mode == "split" else 0
    if expected == 0:
        return ()
    compatible = [decision.faces for decision in decisions if len(decision.faces) == expected]
    if not compatible:
        return ()
    return tuple(
        _median_face([faces[index] for faces in compatible])
        for index in range(expected)
    )


def _sample_evidence_mode(faces: Tuple[FaceBox, ...], frame_width: int) -> str:
    count = len(_normalize_sample_faces(list(faces), frame_width))
    if count == 0:
        return "no-face"
    if count == 1:
        return "single"
    if count == 2:
        return "split"
    return "crowded"


def _earliest_credible_transition_timestamp(
    history: List[TimedFaceObservation],
    target_mode: str,
    frame_width: int,
) -> float:
    """Find the first supporting sample in the sequence later confirmed stable."""
    if not history:
        return 0.0

    earliest: Optional[float] = None
    for observation in reversed(history):
        evidence_mode = _sample_evidence_mode(observation.faces, frame_width)
        supports_target = (
            evidence_mode == target_mode
            or target_mode == "fit-blur" and evidence_mode in ("no-face", "crowded")
        )
        if supports_target:
            earliest = observation.timestamp
            continue

        # A miss is weak evidence while SINGLE/SPLIT establishes; it does not
        # erase an otherwise contiguous supporting sequence.
        if target_mode in ("single", "split") and evidence_mode == "no-face":
            continue
        break
    return earliest if earliest is not None else history[-1].timestamp


def _rolling_layout_blocks(
    observations: List[TimedFaceObservation],
    duration: float,
    frame_width: int,
    config: SmartLayoutTemporalConfig,
) -> List[LayoutBlock]:
    """Convert rolling evidence into confirmed, fixed render-state blocks."""
    if duration <= 0:
        return [LayoutBlock(0.0, max(0.0, duration), "fit-blur")]

    history: List[TimedFaceObservation] = []
    blocks: List[LayoutBlock] = []
    current_mode: Optional[str] = None
    current_faces: Tuple[FaceBox, ...] = ()
    current_start = 0.0
    current_reason = "initial evidence"
    current_cut_search_start: Optional[float] = None
    current_coarse_cut: Optional[float] = None

    candidate_mode: Optional[str] = None
    candidate_start = 0.0
    candidate_decisions: List[FaceDecision] = []
    scene_anchor: Optional[float] = None
    scene_search_start: Optional[float] = None
    force_reblock = False
    ambiguous_run = 0
    ambiguous_start = 0.0
    previous_observation_timestamp: Optional[float] = None

    for observation in observations:
        prior_observation_timestamp = previous_observation_timestamp
        previous_observation_timestamp = observation.timestamp
        if observation.scene_cut:
            history.clear()
            candidate_mode = None
            candidate_decisions = []
            scene_anchor = observation.timestamp
            scene_search_start = prior_observation_timestamp
            force_reblock = current_mode is not None
            ambiguous_run = 0

        history.append(observation)
        cutoff = observation.timestamp - config.rolling_window
        history = [item for item in history if item.timestamp >= cutoff]
        decision = _decide_face_samples(
            [list(item.faces) for item in history],
            frame_width,
        )
        if decision.mode in ("single", "split"):
            target_mode: Optional[str] = decision.mode
            ambiguous_run = 0
        elif decision.mode in ("crowded", "no-face"):
            target_mode = "fit-blur"
            ambiguous_run = 0
        else:
            # Ambiguous evidence maintains the current state and cannot create
            # an immediate transition. Sustained ambiguity eventually becomes
            # the safe fallback instead of preserving stale geometry forever.
            if ambiguous_run == 0:
                ambiguous_start = observation.timestamp
            ambiguous_run += 1
            if ambiguous_run < SMART_LAYOUT_AMBIGUOUS_FALLBACK_SAMPLES:
                candidate_mode = None
                candidate_decisions = []
                continue
            target_mode = "fit-blur"

        needs_transition = current_mode is None or target_mode != current_mode
        if force_reblock and target_mode == current_mode and target_mode in ("single", "split"):
            # A real camera cut may retain the same layout type but needs fresh,
            # fixed geometry for the new shot.
            needs_transition = True
        if not needs_transition:
            candidate_mode = None
            candidate_decisions = []
            scene_anchor = None
            scene_search_start = None
            force_reblock = False
            continue

        if candidate_mode != target_mode:
            candidate_mode = target_mode
            if scene_anchor is not None:
                candidate_start = scene_anchor
            elif ambiguous_run >= SMART_LAYOUT_AMBIGUOUS_FALLBACK_SAMPLES:
                candidate_start = ambiguous_start
            elif current_mode is None:
                candidate_start = 0.0
            else:
                candidate_start = _earliest_credible_transition_timestamp(
                    history,
                    target_mode,
                    frame_width,
                )
            candidate_decisions = [decision]
        else:
            candidate_decisions.append(decision)

        required = (
            SMART_LAYOUT_SCENE_CUT_CONFIRM_SAMPLES
            if scene_anchor is not None
            else SMART_LAYOUT_TRANSITION_CONFIRM_SAMPLES
        )
        if ambiguous_run >= SMART_LAYOUT_AMBIGUOUS_FALLBACK_SAMPLES:
            # The ambiguity run itself is the confirmation evidence.
            candidate_decisions = [decision] * required
        if len(candidate_decisions) < required:
            continue

        transition_start = candidate_start
        if current_mode is not None:
            earliest_transition = current_start + SMART_LAYOUT_MIN_BLOCK_SECONDS
            if observation.timestamp < earliest_transition:
                continue
            transition_start = max(transition_start, earliest_transition)
            blocks.append(
                LayoutBlock(
                    current_start,
                    transition_start,
                    current_mode,
                    current_faces,
                    stabilization=current_reason,
                    cut_search_start=current_cut_search_start,
                    coarse_cut=current_coarse_cut,
                )
            )

        confirmed_scene_cut = (
            scene_anchor is not None
            and abs(transition_start - scene_anchor) < 1e-9
        )
        current_mode = target_mode
        current_faces = _robust_candidate_faces(candidate_decisions, target_mode)
        current_start = transition_start
        current_reason = "scene-cut evidence" if scene_anchor is not None else "sustained evidence"
        current_cut_search_start = scene_search_start if confirmed_scene_cut else None
        current_coarse_cut = scene_anchor if confirmed_scene_cut else None
        candidate_mode = None
        candidate_decisions = []
        scene_anchor = None
        scene_search_start = None
        force_reblock = False

    if current_mode is None:
        return [LayoutBlock(0.0, duration, "fit-blur", stabilization="insufficient evidence")]

    blocks.append(
        LayoutBlock(
            current_start,
            duration,
            current_mode,
            current_faces,
            stabilization=current_reason,
            cut_search_start=current_cut_search_start,
            coarse_cut=current_coarse_cut,
        )
    )
    if len(blocks) > 1 and blocks[-1].end - blocks[-1].start < SMART_LAYOUT_MIN_BLOCK_SECONDS:
        blocks.pop()
        previous = blocks[-1]
        blocks[-1] = LayoutBlock(
            previous.start,
            duration,
            previous.mode,
            previous.faces,
            stabilization=previous.stabilization,
            cut_search_start=previous.cut_search_start,
            coarse_cut=previous.coarse_cut,
        )
    return blocks


def _refine_hard_cut_frame(
    cap,
    cv2,
    fps: float,
    search_start: float,
    coarse_cut: float,
    scene_cut_threshold: float,
) -> Optional[int]:
    """Locate the strongest frame discontinuity inside one confirmed cut bracket."""
    first_frame = max(0, int(round(search_start * fps)))
    last_frame = max(first_frame + 1, int(round(coarse_cut * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    ok, frame = cap.read()
    if not ok:
        return None

    previous_signature = _scene_signature(frame, cv2)
    strongest_score = -1.0
    strongest_frame: Optional[int] = None
    for frame_index in range(first_frame + 1, last_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        signature = _scene_signature(frame, cv2)
        score = _scene_change_score(previous_signature, signature, cv2)
        if score > strongest_score:
            strongest_score = score
            strongest_frame = frame_index
        previous_signature = signature

    if strongest_score < scene_cut_threshold:
        return None
    return strongest_frame


def _refine_and_index_layout_blocks(
    cap,
    cv2,
    blocks: List[LayoutBlock],
    fps: float,
    frame_count: int,
    scene_cut_threshold: float,
) -> List[LayoutBlock]:
    """Refine confirmed scene cuts, then assign gapless shared frame boundaries."""
    refined = list(blocks)
    for index in range(1, len(refined)):
        block = refined[index]
        if block.cut_search_start is None or block.coarse_cut is None:
            continue
        cut_frame = _refine_hard_cut_frame(
            cap,
            cv2,
            fps,
            block.cut_search_start,
            block.coarse_cut,
            scene_cut_threshold,
        )
        if cut_frame is None:
            continue
        refined_time = cut_frame / fps
        if refined_time - refined[index - 1].start < SMART_LAYOUT_MIN_BLOCK_SECONDS:
            continue
        refined[index] = replace(
            block,
            start=refined_time,
            refined_cut=refined_time,
            cut_frame=cut_frame,
        )

    starts = [0]
    for block in refined[1:]:
        starts.append(
            block.cut_frame
            if block.cut_frame is not None
            else max(0, int(round(block.start * fps)))
        )
    total_frames = max(starts[-1], frame_count)
    indexed: List[LayoutBlock] = []
    for index, block in enumerate(refined):
        start_frame = starts[index]
        end_frame = starts[index + 1] if index + 1 < len(starts) else total_frames
        start_time = start_frame / fps
        end_time = end_frame / fps if index + 1 < len(starts) else block.end
        indexed.append(
            replace(
                block,
                start=start_time,
                end=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return indexed


def _analyze_smart_layout(
    cap,
    face_detector,
    cv2,
    fps: float,
    duration: float,
    frame_width: Optional[int] = None,
    temporal_config: Optional[SmartLayoutTemporalConfig] = None,
) -> List[LayoutBlock]:
    """Sample near the timeline, then derive stable blocks from rolling evidence."""
    if duration <= 0 or fps <= 0:
        return [LayoutBlock(0.0, max(0.0, duration), "fit-blur")]

    config = temporal_config or _smart_layout_temporal_config()
    observations: List[TimedFaceObservation] = []
    source_width = frame_width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    previous_signature = None
    timestamp = 0.0
    try:
        while timestamp < duration:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(timestamp * fps))))
            ok, frame = cap.read()
            if ok:
                signature = _scene_signature(frame, cv2)
                scene_cut = (
                    previous_signature is not None
                    and _scene_change_score(previous_signature, signature, cv2)
                    >= config.scene_cut_threshold
                )
                observations.append(
                    TimedFaceObservation(
                        timestamp,
                        tuple(_detect_face_boxes(frame, face_detector, cv2)),
                        scene_cut,
                    )
                )
                previous_signature = signature
            timestamp += config.sample_interval
    finally:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    blocks = _rolling_layout_blocks(observations, duration, source_width, config)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return _refine_and_index_layout_blocks(
        cap,
        cv2,
        blocks,
        fps,
        frame_count,
        config.scene_cut_threshold,
    )


def _clamped_crop(
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
    src_w: int,
    src_h: int,
) -> CropBox:
    """Return an even, source-bounded crop rectangle."""
    crop_w = max(2, min(src_w, crop_w))
    crop_h = max(2, min(src_h, crop_h))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    x = max(0, min(src_w - crop_w, int(round(center_x - crop_w / 2))))
    y = max(0, min(src_h - crop_h, int(round(center_y - crop_h / 2))))
    return x, y, crop_w, crop_h


def _face_crop(
    face: FaceBox,
    src_w: int,
    src_h: int,
    target_ratio: float,
    face_height_fraction: float,
    minimum_source_height_fraction: float,
) -> CropBox:
    """Build a conservative fixed crop with the face above vertical center."""
    x, y, w, h = face
    maximum_h = min(src_h, int(src_w / target_ratio))
    desired_h = int(h / face_height_fraction)
    crop_h = max(int(src_h * minimum_source_height_fraction), desired_h)
    crop_h = min(maximum_h, crop_h)
    crop_w = int(crop_h * target_ratio)
    # Put the face center around 36% down the view, leaving headroom while
    # retaining shoulders/upper torso below it.
    center_x = x + w / 2
    crop_y = y + h / 2 - crop_h * 0.36
    return _clamped_crop(
        center_x,
        crop_y + crop_h / 2,
        crop_w,
        crop_h,
        src_w,
        src_h,
    )


def _single_crop(face: FaceBox, src_w: int, src_h: int) -> CropBox:
    return _face_crop(face, src_w, src_h, 9.0 / 16.0, 0.22, 0.55)


def _split_crop(face: FaceBox, src_w: int, src_h: int) -> CropBox:
    return _face_crop(face, src_w, src_h, 9.0 / 8.0, 0.30, 0.42)


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


def _smart_layout_filter(
    blocks: List[LayoutBlock],
    src_w: int,
    src_h: int,
) -> str:
    """Build fixed per-block layouts and concatenate them with hard cuts."""
    filters: List[str] = []
    outputs: List[str] = []
    for index, block in enumerate(blocks):
        source = f"source{index}"
        output = f"layout{index}"
        if block.start_frame is not None and block.end_frame is not None:
            trim = f"trim=start_frame={block.start_frame}:end_frame={block.end_frame}"
        else:
            trim = f"trim=start={block.start:.6f}:end={block.end:.6f}"
        filters.append(
            f"[0:v]{trim},setpts=PTS-STARTPTS[{source}]"
        )
        if block.mode == "single" and block.faces:
            x, y, width, height = _single_crop(block.faces[0], src_w, src_h)
            filters.append(
                f"[{source}]crop={width}:{height}:{x}:{y},"
                f"scale=1080:1920:flags=lanczos,setsar=1[{output}]"
            )
        elif block.mode == "split" and len(block.faces) == 2:
            left_source = f"left{index}"
            right_source = f"right{index}"
            filters.append(f"[{source}]split=2[{left_source}][{right_source}]")
            half_outputs = []
            for position, (input_label, face) in enumerate(
                ((left_source, block.faces[0]), (right_source, block.faces[1]))
            ):
                x, y, width, height = _split_crop(face, src_w, src_h)
                half_output = f"half{index}_{position}"
                filters.append(
                    f"[{input_label}]crop={width}:{height}:{x}:{y},"
                    f"scale=1080:960:flags=lanczos,setsar=1[{half_output}]"
                )
                half_outputs.append(half_output)
            filters.append(
                f"[{half_outputs[0]}][{half_outputs[1]}]vstack=inputs=2[{output}]"
            )
        else:
            raw_output = f"fallback{index}"
            filters.append(
                _fit_blur_filter_segment(source, raw_output, suffix=str(index))
            )
            filters.append(f"[{raw_output}]setsar=1[{output}]")
        outputs.append(output)

    if len(outputs) == 1:
        filters.append(f"[{outputs[0]}]null[video]")
    else:
        inputs = "".join(f"[{output}]" for output in outputs)
        filters.append(f"{inputs}concat=n={len(outputs)}:v=1:a=0[video]")
    return ";".join(filters)


def _smart_layout_encode_command(
    in_path: str,
    out_path: str,
    blocks: List[LayoutBlock],
    src_w: int,
    src_h: int,
) -> List[str]:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-filter_complex", _smart_layout_filter(blocks, src_w, src_h),
        "-map", "[video]", "-map", "0:a:0?",
    ]
    cmd.extend(_publish_encode_options())
    cmd.extend(["-shortest", out_path])
    return cmd


def _render_smart_layout(
    cap,
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    cv2,
    src_w: int,
    src_h: int,
    fps: float,
) -> str:
    """Analyze then render fixed single/split/fallback blocks with continuous audio."""
    if not _is_9_16(aspect_ratio):
        cap.release()
        raise ValueError("smart-layout currently requires a 9:16 output aspect ratio")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0.0
    temporal_config = _smart_layout_temporal_config()
    print(
        f"[smart-layout] sample_interval={temporal_config.sample_interval:.2f}s "
        f"rolling_window={temporal_config.rolling_window:.2f}s",
        flush=True,
    )
    try:
        face_detector = _create_smart_face_detector(cv2)
        blocks = _analyze_smart_layout(
            cap,
            face_detector,
            cv2,
            fps,
            duration,
            frame_width=src_w,
            temporal_config=temporal_config,
        )
    except Exception as exc:
        # Analysis is advisory: a detector/read failure must produce the safe
        # layout, rather than losing an otherwise renderable highlight.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"[smart-layout] analysis unavailable ({exc}); using fit-blur", flush=True)
        blocks = [
            LayoutBlock(
                0.0,
                duration,
                "fit-blur",
                raw_mode="detector-failure",
            )
        ]
    if not blocks:
        blocks = [LayoutBlock(0.0, duration, "fit-blur", raw_mode="no-face")]
    for index, block in enumerate(blocks):
        if block.coarse_cut is not None and block.refined_cut is not None:
            print(
                f"[smart-layout] cut coarse={block.coarse_cut:.3f}s "
                f"refined={block.refined_cut:.3f}s fps={fps:.3f} "
                f"frame={block.cut_frame}",
                flush=True,
            )
        print(
            f"[smart-layout] {block.start:.2f}-{block.end:.2f}s {block.mode}",
            flush=True,
        )
        if index:
            previous = blocks[index - 1]
            print(
                f"[smart-layout] transition {previous.mode} -> {block.mode} "
                f"at {block.start:.2f}s: {block.stabilization}",
                flush=True,
            )

    cap.release()
    cmd = _smart_layout_encode_command(in_path, out_path, blocks, src_w, src_h)
    subprocess.run(cmd, check=True)
    return out_path


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

    if crop_mode == "smart-layout":
        return _render_smart_layout(
            cap,
            in_path,
            out_path,
            aspect_ratio,
            cv2,
            src_w,
            src_h,
            fps,
        )

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

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
