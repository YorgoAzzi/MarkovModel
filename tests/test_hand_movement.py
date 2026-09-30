import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from hand_movement import (
    ChordGesture,
    FingerStateFilter,
    HandObservation,
    LeftHandControlExtractor,
    Point3D,
    StableChordGesture,
    TiltQualityClassifier,
    chord_degree_for_fingers,
    chord_quality_for_tilt,
    extended_fingers,
)
from hand_performance import parse_args


def hand_with_direction(middle_x, middle_y):
    points = [Point3D(0.5, 0.8) for _ in range(21)]
    points[0] = Point3D(0.5, 0.8)
    points[5] = Point3D(0.42, 0.62)
    points[9] = Point3D(middle_x, middle_y)
    points[13] = Point3D(0.56, 0.62)
    points[17] = Point3D(0.62, 0.66)
    return HandObservation("Left", 1.0, tuple(points))


def hand_with_thumb(thumb_points):
    points = [Point3D(0.0, 0.0) for _ in range(21)]
    points[0] = Point3D(0.00, 0.00)
    points[5] = Point3D(-0.04, 0.08)
    points[9] = Point3D(0.00, 0.09)
    points[13] = Point3D(0.03, 0.08)
    points[17] = Point3D(0.06, 0.06)
    for index, point in zip((1, 2, 3, 4), thumb_points):
        points[index] = Point3D(*point)
    return HandObservation("Right", 1.0, tuple(points), tuple(points))


class GestureMappingTests(unittest.TestCase):
    def test_swapped_hands_are_the_interface_default(self):
        self.assertTrue(parse_args([]).swap_hands)
        self.assertFalse(parse_args(["--no-swap-hands"]).swap_hands)

    def test_requested_finger_combinations_map_to_seven_degrees(self):
        patterns = [
            (False, True, False, False, False),
            (False, True, True, False, False),
            (False, True, True, True, False),
            (False, True, True, True, True),
            (True, True, True, True, True),
            (True, False, False, False, False),
            (True, True, False, False, False),
        ]
        self.assertEqual([chord_degree_for_fingers(value) for value in patterns], list(range(1, 8)))
        self.assertIsNone(chord_degree_for_fingers((False,) * 5))

    def test_tilt_has_minor_diatonic_and_major_zones(self):
        self.assertEqual(chord_quality_for_tilt(-30), "minor")
        self.assertEqual(chord_quality_for_tilt(0), "diatonic")
        self.assertEqual(chord_quality_for_tilt(30), "major")

    def test_quality_hysteresis_prevents_boundary_flicker(self):
        classifier = TiltQualityClassifier(threshold=18, hysteresis=4)
        self.assertEqual(classifier.update(20), "major")
        self.assertEqual(classifier.update(16), "major")
        self.assertEqual(classifier.update(13), "diatonic")

    def test_gesture_requires_dwell_and_reports_only_real_changes(self):
        stabilizer = StableChordGesture(dwell_seconds=0.25)
        gesture = ChordGesture(3, "minor", -25.0, (False, True, True, True, False))
        self.assertEqual(stabilizer.update(gesture, 1.0), (None, False))
        self.assertEqual(stabilizer.update(gesture, 1.2), (None, False))
        accepted, changed = stabilizer.update(gesture, 1.3)
        self.assertTrue(changed)
        self.assertEqual((accepted.degree, accepted.quality), (3, "minor"))
        self.assertFalse(stabilizer.update(gesture, 1.4)[1])

    def test_single_bad_frame_does_not_change_filtered_fingers(self):
        finger_filter = FingerStateFilter(rise_time=0.06, fall_time=0.10)
        index_only = (False, True, False, False, False)
        self.assertEqual(finger_filter.update(index_only, 0.0), index_only)
        self.assertEqual(finger_filter.update((False,) * 5, 0.02), index_only)
        self.assertEqual(finger_filter.update((False,) * 5, 0.30), (False,) * 5)

    def test_thumb_must_be_separated_from_palm(self):
        open_thumb = hand_with_thumb(
            ((-0.01, 0.01), (-0.035, 0.035), (-0.060, 0.060), (-0.090, 0.085))
        )
        tucked_thumb = hand_with_thumb(
            ((-0.01, 0.01), (-0.005, 0.035), (0.002, 0.055), (0.010, 0.075))
        )
        self.assertTrue(extended_fingers(open_thumb)[0])
        self.assertFalse(extended_fingers(tucked_thumb)[0])

    def test_brief_missing_frame_does_not_restart_gesture_dwell(self):
        stabilizer = StableChordGesture(dwell_seconds=0.20, dropout_grace_seconds=0.12)
        gesture = ChordGesture(2, "diatonic", 0.0, (False, True, True, False, False))
        stabilizer.update(gesture, 1.0)
        stabilizer.update(None, 1.05)
        accepted, changed = stabilizer.update(gesture, 1.21)
        self.assertTrue(changed)
        self.assertEqual(accepted.degree, 2)


class LeftHandTests(unittest.TestCase):
    def test_height_controls_volume_and_left_tilt_controls_density(self):
        extractor = LeftHandControlExtractor(
            max_left_tilt=45.0,
            tilt_dead_zone=5.0,
            tilt_time_constant=0.01,
        )
        upright = extractor.update(hand_with_direction(0.5, 0.5), 0.0)
        tilted_left = extractor.update(hand_with_direction(0.3, 0.65), 0.1)
        tilted_right = extractor.update(hand_with_direction(0.7, 0.65), 0.2)
        self.assertGreater(tilted_left.density, upright.density)
        self.assertLess(tilted_right.density, tilted_left.density)
        self.assertGreaterEqual(tilted_left.density, 0.0)
        self.assertLessEqual(tilted_left.density, 1.0)
        self.assertGreater(upright.volume, 0.0)


if __name__ == "__main__":
    unittest.main()
