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

    def test_all_existing_crop_modes_and_smart_layout_validate(self) -> None:
        self.assertEqual(
            [
                clipper._validate_crop_mode(mode)
                for mode in ("center", "static-face", "face", "fit-blur", "smart-layout")
            ],
            ["center", "static-face", "face", "fit-blur", "smart-layout"],
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


class SmartLayoutDecisionTests(unittest.TestCase):
    LEFT = (100, 80, 80, 80)
    RIGHT = (700, 90, 75, 75)
    THIRD = (400, 70, 70, 70)

    def test_one_stable_face_selects_single(self) -> None:
        mode, faces = clipper._classify_face_samples([[self.LEFT] for _ in range(7)])

        self.assertEqual(mode, "single")
        self.assertEqual(faces, (self.LEFT,))

    def test_two_stable_faces_select_split_left_to_right(self) -> None:
        samples = [[self.RIGHT, self.LEFT] for _ in range(7)]
        samples = [sorted(faces, key=lambda face: face[0]) for faces in samples]

        mode, faces = clipper._classify_face_samples(samples)

        self.assertEqual(mode, "split")
        self.assertLess(faces[0][0], faces[1][0])

    def test_duplicate_detections_of_one_person_never_select_split(self) -> None:
        person = (760, 180, 180, 180)
        duplicate = (775, 190, 170, 170)
        samples = [
            [person, duplicate],
            [person],
            [duplicate, person],
            [person],
            [person, duplicate],
            [person],
            [duplicate, person],
        ]

        mode, faces = clipper._classify_face_samples(
            samples,
            frame_width=1920,
        )

        self.assertEqual(mode, "single")
        self.assertEqual(len(faces), 1)

    def test_nearby_nonoverlapping_detections_are_not_distinct_people(self) -> None:
        first = (700, 180, 140, 140)
        nearby = (900, 185, 135, 135)

        mode, faces = clipper._classify_face_samples(
            [[first, nearby] for _ in range(7)],
            frame_width=1920,
        )

        self.assertEqual(mode, "single")
        self.assertEqual(len(faces), 1)

    def test_two_real_people_with_one_missed_face_still_select_split(self) -> None:
        samples = [[self.LEFT, self.RIGHT] for _ in range(6)] + [[self.LEFT]]

        self.assertEqual(
            clipper._classify_face_samples(samples, frame_width=1000)[0],
            "split",
        )

    def test_no_faces_uses_fit_blur(self) -> None:
        self.assertEqual(
            clipper._classify_face_samples([[] for _ in range(7)]),
            ("fit-blur", ()),
        )

    def test_three_or_more_faces_uses_fit_blur(self) -> None:
        crowded = [[self.LEFT, self.THIRD, self.RIGHT] for _ in range(7)]

        self.assertEqual(clipper._classify_face_samples(crowded), ("fit-blur", ()))

    def test_one_bad_frame_does_not_change_single_layout(self) -> None:
        samples = [[self.LEFT], [self.LEFT], [], [self.LEFT], [self.LEFT], [self.LEFT], [self.LEFT]]

        self.assertEqual(clipper._classify_face_samples(samples)[0], "single")

    def test_split_order_uses_median_left_and_right_identity(self) -> None:
        samples = [
            [(105, 82, 80, 80), (705, 92, 75, 75)],
            [(95, 78, 80, 80), (695, 88, 75, 75)],
            [self.LEFT, self.RIGHT],
            [self.LEFT, self.RIGHT],
            [self.LEFT, self.RIGHT],
        ]

        mode, faces = clipper._classify_face_samples(samples)

        self.assertEqual(mode, "split")
        self.assertEqual(faces, (self.LEFT, self.RIGHT))

    def test_short_tail_is_absorbed_to_enforce_minimum_hold(self) -> None:
        blocks = [
            clipper.LayoutBlock(0.0, 3.0, "single", (self.LEFT,)),
            clipper.LayoutBlock(3.0, 5.0, "split", (self.LEFT, self.RIGHT)),
        ]

        merged = clipper._merge_layout_blocks(blocks, min_hold=2.5)

        self.assertEqual(merged, [clipper.LayoutBlock(0.0, 5.0, "single", (self.LEFT,))])

    def test_single_bridges_one_noisy_middle_window(self) -> None:
        raw = [
            clipper.LayoutBlock(0.0, 3.0, "single", (self.LEFT,), people=1),
            clipper.LayoutBlock(
                3.0, 6.0, "ambiguous", (self.LEFT,), people=1
            ),
            clipper.LayoutBlock(6.0, 9.0, "single", (self.LEFT,), people=1),
        ]

        stabilized = clipper._stabilize_layout_blocks(raw)

        self.assertEqual([block.mode for block in stabilized], ["single"] * 3)
        self.assertEqual(stabilized[1].stabilization, "carry-forward")

    def test_split_bridges_one_noisy_middle_window(self) -> None:
        pair = (self.LEFT, self.RIGHT)
        raw = [
            clipper.LayoutBlock(0.0, 3.0, "split", pair, people=2),
            clipper.LayoutBlock(3.0, 6.0, "ambiguous", pair, people=2),
            clipper.LayoutBlock(6.0, 9.0, "split", pair, people=2),
        ]

        stabilized = clipper._stabilize_layout_blocks(raw)

        self.assertEqual([block.mode for block in stabilized], ["split"] * 3)

    def test_sustained_no_faces_eventually_falls_back(self) -> None:
        raw = [
            clipper.LayoutBlock(0.0, 3.0, "single", (self.LEFT,), people=1),
            clipper.LayoutBlock(3.0, 6.0, "no-face", people=0),
            clipper.LayoutBlock(6.0, 9.0, "no-face", people=0),
        ]

        stabilized = clipper._stabilize_layout_blocks(raw)

        self.assertEqual(
            [block.mode for block in stabilized],
            ["single", "single", "fit-blur"],
        )

    def test_two_weak_windows_are_not_all_bridged_by_later_single(self) -> None:
        raw = [
            clipper.LayoutBlock(0.0, 3.0, "single", (self.LEFT,), people=1),
            clipper.LayoutBlock(3.0, 6.0, "no-face", people=0),
            clipper.LayoutBlock(6.0, 9.0, "no-face", people=0),
            clipper.LayoutBlock(9.0, 12.0, "single", (self.LEFT,), people=1),
        ]

        self.assertEqual(
            [block.mode for block in clipper._stabilize_layout_blocks(raw)],
            ["single", "single", "fit-blur", "single"],
        )

    def test_genuine_layout_change_switches_immediately(self) -> None:
        raw = [
            clipper.LayoutBlock(0.0, 3.0, "single", (self.LEFT,), people=1),
            clipper.LayoutBlock(
                3.0, 6.0, "split", (self.LEFT, self.RIGHT), people=2
            ),
        ]

        self.assertEqual(
            [block.mode for block in clipper._stabilize_layout_blocks(raw)],
            ["single", "split"],
        )

    def test_distant_single_subjects_keep_separate_fixed_blocks(self) -> None:
        blocks = [
            clipper.LayoutBlock(0.0, 3.0, "single", (self.LEFT,)),
            clipper.LayoutBlock(3.0, 6.0, "single", (self.RIGHT,)),
        ]

        self.assertEqual(clipper._merge_layout_blocks(blocks), blocks)

    def test_single_and_split_crops_stay_inside_source(self) -> None:
        source_width, source_height = 1920, 1080
        edge_faces = [(0, 0, 90, 90), (1830, 990, 90, 90)]

        for face in edge_faces:
            for crop in (
                clipper._single_crop(face, source_width, source_height),
                clipper._split_crop(face, source_width, source_height),
            ):
                x, y, width, height = crop
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + width, source_width)
                self.assertLessEqual(y + height, source_height)
                self.assertEqual(width % 2, 0)
                self.assertEqual(height % 2, 0)


class SmartLayoutDetectorTests(unittest.TestCase):
    def test_yunet_initialization_is_centralized_and_uses_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory) / "yunet.onnx"
            model.touch()
            fake_cv2 = types.SimpleNamespace(FaceDetectorYN_create=Mock(return_value="detector"))

            detector = clipper._create_smart_face_detector(
                fake_cv2,
                model_path=str(model),
            )

        self.assertEqual(detector, "detector")
        fake_cv2.FaceDetectorYN_create.assert_called_once_with(
            str(model),
            "",
            (320, 320),
            clipper.SMART_YUNET_CONFIDENCE,
            clipper.SMART_YUNET_NMS_THRESHOLD,
            clipper.SMART_YUNET_TOP_K,
        )

    def test_missing_yunet_model_reports_configuration_path(self) -> None:
        fake_cv2 = types.SimpleNamespace(FaceDetectorYN_create=Mock())

        with self.assertRaisesRegex(RuntimeError, "SMART_LAYOUT_YUNET_MODEL"):
            clipper._create_smart_face_detector(
                fake_cv2,
                model_path="/definitely/missing/yunet.onnx",
            )

        fake_cv2.FaceDetectorYN_create.assert_not_called()


class SmartLayoutWindowConfigTests(unittest.TestCase):
    def test_default_window_remains_three_seconds(self) -> None:
        with patch.dict(clipper.os.environ, {}, clear=False):
            clipper.os.environ.pop("SMART_LAYOUT_WINDOW_SECONDS", None)

            self.assertEqual(clipper._smart_layout_window_seconds(), 3.0)

    def test_environment_override_uses_two_seconds(self) -> None:
        with patch.dict(
            clipper.os.environ,
            {"SMART_LAYOUT_WINDOW_SECONDS": "2.0"},
        ):
            self.assertEqual(clipper._smart_layout_window_seconds(), 2.0)

    def test_invalid_window_values_raise_clear_error(self) -> None:
        for value in ("not-a-number", "0", "0.49", "inf", "31"):
            with self.subTest(value=value), patch.dict(
                clipper.os.environ,
                {"SMART_LAYOUT_WINDOW_SECONDS": value},
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "SMART_LAYOUT_WINDOW_SECONDS",
                ):
                    clipper._smart_layout_window_seconds()

    def test_legacy_fixed_window_does_not_change_rolling_sampling(self) -> None:
        capture = FakeSamplingCapture(frame_count=100)
        fake_cv2 = types.SimpleNamespace(
            CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_POS_FRAMES=2,
            CAP_PROP_FRAME_COUNT=3,
        )
        face = (100, 80, 80, 80)
        temporal_config = clipper.SmartLayoutTemporalConfig(0.25, 1.25, 0.22)

        with (
            patch.dict(
                clipper.os.environ,
                {"SMART_LAYOUT_WINDOW_SECONDS": "2.0"},
            ),
            patch.object(clipper, "_detect_face_boxes", return_value=[face]),
            patch.object(clipper, "_scene_signature", return_value=object()),
            patch.object(clipper, "_scene_change_score", return_value=0.0),
        ):
            blocks = clipper._analyze_smart_layout(
                capture,
                Mock(),
                fake_cv2,
                fps=10.0,
                duration=1.0,
                frame_width=1000,
                temporal_config=temporal_config,
            )

        self.assertEqual([(block.start, block.end) for block in blocks], [(0.0, 1.0)])
        self.assertEqual(blocks[0].mode, "single")
        self.assertEqual(capture.position, 0)


class SmartLayoutRollingConfigTests(unittest.TestCase):
    def test_rolling_defaults(self) -> None:
        names = (
            "SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS",
            "SMART_LAYOUT_ROLLING_WINDOW_SECONDS",
            "SMART_LAYOUT_SCENE_CUT_THRESHOLD",
        )
        with patch.dict(clipper.os.environ, {}, clear=False):
            for name in names:
                clipper.os.environ.pop(name, None)

            config = clipper._smart_layout_temporal_config()

        self.assertEqual(config.sample_interval, 0.25)
        self.assertEqual(config.rolling_window, 1.25)
        self.assertEqual(config.scene_cut_threshold, 0.22)

    def test_rolling_environment_overrides(self) -> None:
        with patch.dict(
            clipper.os.environ,
            {
                "SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS": "0.20",
                "SMART_LAYOUT_ROLLING_WINDOW_SECONDS": "1.0",
                "SMART_LAYOUT_SCENE_CUT_THRESHOLD": "0.30",
            },
        ):
            config = clipper._smart_layout_temporal_config()

        self.assertEqual(config, clipper.SmartLayoutTemporalConfig(0.2, 1.0, 0.3))

    def test_rolling_configuration_rejects_invalid_values(self) -> None:
        cases = [
            ("SMART_LAYOUT_SAMPLE_INTERVAL_SECONDS", "0"),
            ("SMART_LAYOUT_ROLLING_WINDOW_SECONDS", "invalid"),
            ("SMART_LAYOUT_SCENE_CUT_THRESHOLD", "1.0"),
        ]
        for name, value in cases:
            with self.subTest(name=name), patch.dict(
                clipper.os.environ,
                {name: value},
            ):
                with self.assertRaisesRegex(ValueError, name):
                    clipper._smart_layout_temporal_config()


class SmartLayoutRollingStateTests(unittest.TestCase):
    LEFT = (100, 80, 80, 80)
    RIGHT = (700, 90, 75, 75)
    THIRD = (400, 70, 70, 70)
    CONFIG = clipper.SmartLayoutTemporalConfig(0.25, 1.25, 0.22)

    def observations(self, duration, faces_at, scene_cut_at=None):
        result = []
        for index in range(int(duration / self.CONFIG.sample_interval)):
            timestamp = index * self.CONFIG.sample_interval
            result.append(
                clipper.TimedFaceObservation(
                    timestamp,
                    tuple(faces_at(timestamp, index)),
                    scene_cut=timestamp == scene_cut_at,
                )
            )
        return result

    def layouts(self, duration, faces_at, scene_cut_at=None):
        return clipper._rolling_layout_blocks(
            self.observations(duration, faces_at, scene_cut_at),
            duration,
            1000,
            self.CONFIG,
        )

    def test_stable_single_ignores_occasional_misses(self) -> None:
        blocks = self.layouts(
            4.0,
            lambda _time, index: [] if index in (5, 11) else [self.LEFT],
        )

        self.assertEqual([(block.start, block.end, block.mode) for block in blocks], [(0.0, 4.0, "single")])

    def test_stable_split_ignores_occasional_single_samples(self) -> None:
        blocks = self.layouts(
            4.0,
            lambda _time, index: [self.LEFT] if index in (6, 12) else [self.LEFT, self.RIGHT],
        )

        self.assertEqual([(block.start, block.end, block.mode) for block in blocks], [(0.0, 4.0, "split")])

    def test_crowd_hard_cut_to_single_transitions_at_cut(self) -> None:
        cut = 1.75
        blocks = self.layouts(
            4.0,
            lambda timestamp, _index: (
                [self.LEFT, self.THIRD, self.RIGHT]
                if timestamp < cut
                else [self.LEFT]
            ),
            scene_cut_at=cut,
        )

        self.assertEqual([block.mode for block in blocks], ["fit-blur", "single"])
        self.assertEqual(blocks[1].start, cut)
        self.assertLess(blocks[1].start, 2.0)
        self.assertEqual(blocks[1].cut_search_start, 1.5)
        self.assertEqual(blocks[1].coarse_cut, cut)

    def test_confirmed_transition_without_scene_cut_backdates_to_first_evidence(self) -> None:
        first_single = 1.75
        blocks = self.layouts(
            4.0,
            lambda timestamp, _index: (
                [self.LEFT, self.THIRD, self.RIGHT]
                if timestamp < first_single
                else [self.LEFT]
            ),
        )

        self.assertEqual([block.mode for block in blocks], ["fit-blur", "single"])
        self.assertEqual(blocks[1].start, first_single)

    def test_unconfirmed_single_transient_is_never_backdated(self) -> None:
        blocks = self.layouts(
            4.0,
            lambda _timestamp, index: (
                [self.LEFT]
                if index in (7, 8)
                else [self.LEFT, self.THIRD, self.RIGHT]
            ),
        )

        self.assertEqual([(block.start, block.end, block.mode) for block in blocks], [(0.0, 4.0, "fit-blur")])

    def test_single_hard_cut_to_split_transitions_at_cut(self) -> None:
        cut = 1.75
        blocks = self.layouts(
            4.0,
            lambda timestamp, _index: (
                [self.LEFT] if timestamp < cut else [self.LEFT, self.RIGHT]
            ),
            scene_cut_at=cut,
        )

        self.assertEqual([block.mode for block in blocks], ["single", "split"])
        self.assertEqual(blocks[1].start, cut)

    def test_same_mode_camera_cut_backdates_new_fixed_geometry(self) -> None:
        cut = 1.75
        blocks = self.layouts(
            4.0,
            lambda timestamp, _index: [self.LEFT] if timestamp < cut else [self.RIGHT],
            scene_cut_at=cut,
        )

        self.assertEqual([block.mode for block in blocks], ["single", "single"])
        self.assertEqual(blocks[1].start, cut)
        self.assertEqual(blocks[0].faces, (self.LEFT,))
        self.assertEqual(blocks[1].faces, (self.RIGHT,))
        self.assertTrue(
            all(
                block.end - block.start >= clipper.SMART_LAYOUT_MIN_BLOCK_SECONDS
                for block in blocks
            )
        )

    def test_normal_motion_does_not_create_same_mode_reblocks(self) -> None:
        blocks = self.layouts(
            4.0,
            lambda _time, index: [(100 + index * 2, 80, 80, 80)],
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].mode, "single")

    def test_short_false_second_face_does_not_create_split_microblock(self) -> None:
        blocks = self.layouts(
            4.0,
            lambda _time, index: (
                [self.LEFT, self.RIGHT] if index == 8 else [self.LEFT]
            ),
        )

        self.assertEqual([block.mode for block in blocks], ["single"])

    def test_sustained_no_face_eventually_transitions_to_fit_blur(self) -> None:
        blocks = self.layouts(
            5.0,
            lambda timestamp, _index: [self.LEFT] if timestamp < 2.0 else [],
        )

        self.assertEqual([block.mode for block in blocks], ["single", "fit-blur"])
        self.assertGreaterEqual(blocks[1].start, 2.0)

    def test_sustained_ambiguous_geometry_eventually_uses_fit_blur(self) -> None:
        blocks = self.layouts(
            6.0,
            lambda timestamp, index: (
                [self.LEFT]
                if timestamp < 2.0
                else [self.LEFT if index % 2 else self.RIGHT]
            ),
        )

        self.assertEqual(blocks[0].mode, "single")
        self.assertEqual(blocks[-1].mode, "fit-blur")

    def test_persistent_crowd_is_fit_blur(self) -> None:
        blocks = self.layouts(
            3.0,
            lambda _time, _index: [self.LEFT, self.THIRD, self.RIGHT],
        )

        self.assertEqual([block.mode for block in blocks], ["fit-blur"])

    def test_scene_metric_is_conservative_for_small_change(self) -> None:
        import cv2

        base = np.full((90, 160), 100, dtype=np.uint8)
        small_change = np.full((90, 160), 110, dtype=np.uint8)
        hard_cut = np.full((90, 160), 255, dtype=np.uint8)

        self.assertLess(clipper._scene_change_score(base, small_change, cv2), 0.22)
        self.assertGreater(clipper._scene_change_score(base, hard_cut, cv2), 0.22)


class SmartLayoutFrameBoundaryTests(unittest.TestCase):
    def test_confirmed_cut_refines_to_exact_source_frame(self) -> None:
        import cv2

        fps = 30.0
        cut_frame = 30
        frames = [
            np.full((90, 160, 3), 20 if index < cut_frame else 240, dtype=np.uint8)
            for index in range(60)
        ]

        class FrameCapture:
            def __init__(self):
                self.position = 0

            def set(self, _property_id, value):
                self.position = int(value)

            def read(self):
                if self.position >= len(frames):
                    return False, None
                frame = frames[self.position]
                self.position += 1
                return True, frame

        capture = FrameCapture()

        refined_frame = clipper._refine_hard_cut_frame(
            capture,
            cv2,
            fps,
            search_start=0.75,
            coarse_cut=1.25,
            scene_cut_threshold=0.22,
        )

        self.assertEqual(refined_frame, cut_frame)

    def test_adjacent_blocks_share_exact_exclusive_frame_boundary(self) -> None:
        cut_frame = 30
        blocks = [
            clipper.LayoutBlock(
                0.0,
                1.0,
                "fit-blur",
                start_frame=0,
                end_frame=cut_frame,
            ),
            clipper.LayoutBlock(
                1.0,
                2.0,
                "single",
                ((40, 20, 30, 30),),
                start_frame=cut_frame,
                end_frame=60,
            ),
        ]

        graph = clipper._smart_layout_filter(blocks, 160, 90)

        self.assertIn("trim=start_frame=0:end_frame=30", graph)
        self.assertIn("trim=start_frame=30:end_frame=60", graph)
        self.assertNotIn("trim=start=", graph)


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

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg and ffprobe are required",
    )
    def test_smart_layout_vertical_render_is_1080x1920_h264_aac(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv-python is required")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            output = root / "smart-layout.mp4"
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
                crop_mode="smart-layout",
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
    def test_smart_layout_mixed_single_split_and_fallback_blocks_encode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            output = root / "mixed.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:r=4:d=1.5",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(source),
                ],
                check=True,
            )
            left = (40, 30, 30, 30)
            right = (240, 30, 30, 30)
            blocks = [
                clipper.LayoutBlock(0.0, 0.5, "single", (left,)),
                clipper.LayoutBlock(0.5, 1.0, "split", (left, right)),
                clipper.LayoutBlock(1.0, 1.5, "fit-blur"),
            ]

            subprocess.run(
                clipper._smart_layout_encode_command(
                    str(source), str(output), blocks, 320, 180
                ),
                check=True,
            )
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height,pix_fmt",
                    "-of", "json", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            video = json.loads(probe.stdout)["streams"][0]

            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(video["width"], 1080)
            self.assertEqual(video["height"], 1920)
            self.assertEqual(video["pix_fmt"], "yuv420p")

    @unittest.skipUnless(
        shutil.which("ffmpeg"),
        "ffmpeg is required",
    )
    def test_frame_indexed_render_has_no_cut_overlap_or_gap(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv-python is required")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=red:s=160x90:r=30:d=1",
                    "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=30:d=1",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[video]",
                    "-map", "[video]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(source),
                ],
                check=True,
            )
            blocks = [
                clipper.LayoutBlock(
                    0.0, 1.0, "fit-blur", start_frame=0, end_frame=30
                ),
                clipper.LayoutBlock(
                    1.0, 2.0, "fit-blur", start_frame=30, end_frame=60
                ),
            ]

            subprocess.run(
                clipper._smart_layout_encode_command(
                    str(source), str(output), blocks, 160, 90
                ),
                check=True,
            )

            capture = cv2.VideoCapture(str(output))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 60)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 29)
            ok_before, before = capture.read()
            ok_after, after = capture.read()
            capture.release()

            self.assertTrue(ok_before)
            self.assertTrue(ok_after)
            self.assertGreater(float(before[:, :, 2].mean()), float(before[:, :, 0].mean()))
            self.assertGreater(float(after[:, :, 0].mean()), float(after[:, :, 2].mean()))


if __name__ == "__main__":
    unittest.main()
