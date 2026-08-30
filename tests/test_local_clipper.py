from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from shorts_generator.local import clipper
from shorts_generator.local.clipper import _final_encode_command, crop_clip_local


class FakeSamplingCapture:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.position = 0

    def get(self, property_id: int) -> int:
        return self.frame_count

    def set(self, property_id: int, value: int) -> None:
        self.position = int(value)

    def read(self):
        frame = self.position
        self.position += 1
        return True, frame


class CropModeTests(unittest.TestCase):
    def test_static_face_uses_median_sampled_face_x(self) -> None:
        capture = FakeSamplingCapture(frame_count=10)
        fake_cv2 = types.SimpleNamespace(CAP_PROP_FRAME_COUNT=1, CAP_PROP_POS_FRAMES=2)
        detected_centers = [(80, 20), None, (20, 20), (50, 20), (110, 20)]

        with patch.object(
            clipper,
            "_primary_face_center",
            side_effect=detected_centers,
        ) as detect:
            center_x = clipper._static_face_center_x(
                capture,
                Mock(),
                src_w=200,
                cv2=fake_cv2,
                sample_count=5,
            )

        self.assertEqual(center_x, 65)
        self.assertEqual(detect.call_count, 5)
        self.assertEqual(capture.position, 0)

    def test_static_face_without_detection_falls_back_to_center(self) -> None:
        capture = FakeSamplingCapture(frame_count=5)
        fake_cv2 = types.SimpleNamespace(CAP_PROP_FRAME_COUNT=1, CAP_PROP_POS_FRAMES=2)

        with patch.object(clipper, "_primary_face_center", return_value=None):
            center_x = clipper._static_face_center_x(
                capture,
                Mock(),
                src_w=200,
                cv2=fake_cv2,
                sample_count=5,
            )

        self.assertEqual(center_x, 100)
        self.assertEqual(capture.position, 0)

    def test_static_face_selects_once_and_crop_does_not_move(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                x_values = np.arange(100, dtype=np.uint8)
                frame = np.broadcast_to(x_values[None, :, None], (50, 100, 3)).copy()
                self.frames = [frame.copy() for _ in range(3)]
                self.position = 0

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                return {1: 100, 2: 50, 3: 5, 4: len(self.frames)}[property_id]

            def set(self, property_id: int, value: int) -> None:
                self.position = int(value)

            def read(self):
                if self.position >= len(self.frames):
                    return False, None
                frame = self.frames[self.position]
                self.position += 1
                return True, frame

            def release(self) -> None:
                pass

        class FakeWriter:
            def __init__(self) -> None:
                self.frames = []

            def write(self, frame) -> None:
                self.frames.append(frame.copy())

            def release(self) -> None:
                pass

        capture = FakeCapture()
        writer = FakeWriter()
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.CAP_PROP_FRAME_WIDTH = 1
        fake_cv2.CAP_PROP_FRAME_HEIGHT = 2
        fake_cv2.CAP_PROP_FPS = 3
        fake_cv2.CAP_PROP_FRAME_COUNT = 4
        fake_cv2.CAP_PROP_POS_FRAMES = 5
        fake_cv2.data = types.SimpleNamespace(haarcascades="")
        fake_cv2.VideoCapture = Mock(return_value=capture)
        fake_cv2.CascadeClassifier = Mock(return_value=Mock())
        fake_cv2.VideoWriter_fourcc = Mock(return_value=0)
        fake_cv2.VideoWriter = Mock(return_value=writer)

        with (
            patch.dict(sys.modules, {"cv2": fake_cv2}),
            patch.object(clipper, "_static_face_center_x", return_value=80) as select,
            patch.object(clipper.subprocess, "run"),
            patch.object(clipper.os, "remove"),
        ):
            clipper._reframe_vertical("cut.mp4", "short.mp4", "2:5", "static-face")

        select.assert_called_once()
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual([int(frame[0, 0, 0]) for frame in writer.frames], [70, 70, 70])

    def test_all_crop_modes_include_existing_dynamic_face_mode(self) -> None:
        self.assertEqual(
            [
                clipper._validate_crop_mode(mode)
                for mode in ("center", "static-face", "face", "fit-blur")
            ],
            ["center", "static-face", "face", "fit-blur"],
        )

    def test_fit_blur_does_not_initialize_face_detection(self) -> None:
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.CascadeClassifier = Mock(
            side_effect=AssertionError("face detector must not be initialized")
        )

        with (
            patch.dict(sys.modules, {"cv2": fake_cv2}),
            patch.object(clipper.subprocess, "run") as run,
        ):
            result = clipper._reframe_vertical(
                "cut.mp4",
                "short.mp4",
                "9:16",
                "fit-blur",
            )

        self.assertEqual(result, "short.mp4")
        fake_cv2.CascadeClassifier.assert_not_called()
        run.assert_called_once()

    def test_fit_blur_foreground_is_contained_without_crop_or_stretch(self) -> None:
        filter_graph = clipper._fit_blur_filter()
        foreground = filter_graph.rsplit("[foreground]", 1)[1].split("[fitted]", 1)[0]

        self.assertIn("scale=1080:1920", foreground)
        self.assertIn("force_original_aspect_ratio=decrease", foreground)
        self.assertNotIn("crop=", foreground)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", filter_graph)

    def test_fit_blur_command_keeps_publish_encoding(self) -> None:
        command = clipper._fit_blur_encode_command("cut.mp4", "short.mp4", "9:16")

        self.assertIn("crop=1080:1920", command[command.index("-filter_complex") + 1])
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-crf") + 1], "18")
        self.assertEqual(command[command.index("-preset") + 1], "fast")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-b:a") + 1], "192k")
        self.assertEqual(command[command.index("-movflags") + 1], "+faststart")


class FinalEncodeCommandTests(unittest.TestCase):
    def test_vertical_output_is_publish_ready_h264_aac(self) -> None:
        command = _final_encode_command(
            "reframed.silent.mp4",
            "cut.mp4",
            "short.mp4",
            "9:16",
        )

        self.assertIn("scale=1080:1920:flags=lanczos", command)
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-preset") + 1], "fast")
        self.assertEqual(command[command.index("-crf") + 1], "18")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-b:a") + 1], "192k")
        self.assertEqual(command[command.index("-movflags") + 1], "+faststart")
        self.assertNotIn("copy", command)
        self.assertIn("-shortest", command)

    def test_non_vertical_output_is_not_resized(self) -> None:
        command = _final_encode_command(
            "reframed.silent.mp4",
            "cut.mp4",
            "short.mp4",
            "1:1",
        )

        self.assertNotIn("-vf", command)
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")


class SyntheticRenderTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg and ffprobe are required",
    )
    def test_vertical_render_streams_are_publish_ready(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv-python is required")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            output = root / "short.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=5:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(source),
                ],
                check=True,
            )

            crop_clip_local(
                str(source),
                0.1,
                0.8,
                "9:16",
                str(output),
                crop_mode="static-face",
            )

            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_streams",
                    "-of", "json", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout)["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            audio = next(stream for stream in streams if stream["codec_type"] == "audio")

            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(video["width"], 1080)
            self.assertEqual(video["height"], 1920)
            self.assertEqual(video["pix_fmt"], "yuv420p")
            self.assertEqual(audio["codec_name"], "aac")

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg and ffprobe are required",
    )
    def test_fit_blur_vertical_render_is_1080x1920_h264_aac(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            output = root / "fit-blur.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:r=3:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(source),
                ],
                check=True,
            )

            crop_clip_local(
                str(source),
                0.1,
                0.8,
                "9:16",
                str(output),
                crop_mode="fit-blur",
            )

            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_streams",
                    "-of", "json", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout)["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            audio = next(stream for stream in streams if stream["codec_type"] == "audio")

            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(video["width"], 1080)
            self.assertEqual(video["height"], 1920)
            self.assertEqual(video["pix_fmt"], "yuv420p")
            self.assertEqual(audio["codec_name"], "aac")


if __name__ == "__main__":
    unittest.main()
