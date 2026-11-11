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
    HandAccelerationPeakDetector,
    HandMelodyHeight,
    TiltQualityClassifier,
    chord_degree_for_fingers,
    chord_quality_for_tilt,
    extended_fingers,
    finger_extension_amount,
    hand_openness,
    height_to_register,
    is_pinching,
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


def shifted_hand(hand, dx=0.0, dy=0.0):
    points = tuple(Point3D(point.x + dx, point.y + dy, point.z) for point in hand.landmarks)
    world = tuple(hand.world_landmarks)
    return HandObservation(hand.handedness, hand.confidence, points, world)


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
        self.assertGreater(hand_openness(open_thumb), hand_openness(tucked_thumb))

    def test_index_extension_is_continuous_and_rotation_independent(self):
        base = hand_with_direction(0.5, 0.5)
        straight_points = list(base.landmarks)
        for index, y in zip((5, 6, 7, 8), (0.60, 0.50, 0.40, 0.30)):
            straight_points[index] = Point3D(0.42, y)
        curled_points = list(straight_points)
        curled_points[7] = Point3D(0.47, 0.52)
        curled_points[8] = Point3D(0.42, 0.57)
        straight = HandObservation("Right", 1.0, tuple(straight_points))
        curled = HandObservation("Right", 1.0, tuple(curled_points))
        self.assertGreater(finger_extension_amount(straight), 0.9)
        self.assertLess(finger_extension_amount(curled), 0.5)

    def test_brief_missing_frame_does_not_restart_gesture_dwell(self):
        stabilizer = StableChordGesture(dwell_seconds=0.20, dropout_grace_seconds=0.12)
        gesture = ChordGesture(2, "diatonic", 0.0, (False, True, True, False, False))
        stabilizer.update(gesture, 1.0)
        stabilizer.update(None, 1.05)
        accepted, changed = stabilizer.update(gesture, 1.21)
        self.assertTrue(changed)
        self.assertEqual(accepted.degree, 2)


class LeftHandTests(unittest.TestCase):
    def test_height_controls_activity_and_slow_horizontal_speed_controls_tempo(self):
        extractor = LeftHandControlExtractor(
            min_tempo=60.0,
            max_tempo=140.0,
            max_horizontal_speed=0.20,
            speed_dead_zone=0.005,
            speed_time_constant=0.01,
        )
        base_hand = hand_with_direction(0.5, 0.5)
        stationary = extractor.update(base_hand, 0.0)
        moving = extractor.update(shifted_hand(base_hand, dx=0.02), 0.1)
        high = extractor.update(shifted_hand(base_hand, dx=0.02, dy=-0.25), 0.2)
        self.assertEqual(stationary.tempo_bpm, 60.0)
        self.assertGreater(moving.tempo_bpm, stationary.tempo_bpm)
        self.assertGreater(high.melodic_activity, moving.melodic_activity)
        self.assertGreaterEqual(moving.articulation, 0.0)
        self.assertLessEqual(moving.articulation, 1.0)

    def test_right_hand_height_maps_to_register(self):
        self.assertEqual(height_to_register(0.25), 1.0)
        self.assertEqual(height_to_register(0.50), 0.0)
        self.assertEqual(height_to_register(0.75), -1.0)
        self.assertAlmostEqual(height_to_register(0.35), 0.6)

    def test_pinch_uses_thumb_index_distance_relative_to_palm(self):
        base = hand_with_direction(0.5, 0.5)
        def with_tip_distance(distance):
            points = list(base.landmarks)
            points[4] = Point3D(0.50, 0.40)
            points[8] = Point3D(0.50 + distance, 0.40)
            return HandObservation("Right", 1.0, tuple(points))

        self.assertTrue(is_pinching(with_tip_distance(0.03), threshold=0.42))
        self.assertFalse(is_pinching(with_tip_distance(0.20), threshold=0.42))


class ConductorMovementTests(unittest.TestCase):
    @staticmethod
    def _peak_count(detector, positions):
        base = hand_with_direction(0.5, 0.5)
        return sum(
            detector.update(
                shifted_hand(base, dx=x - 0.5, dy=y - 0.5),
                index * 0.033,
            ).peak
            for index, (x, y) in enumerate(positions)
        )

    @staticmethod
    def _detector(**overrides):
        settings = dict(
            acceleration_threshold=1.0,
            minimum_speed=0.05,
            velocity_time_constant=0.02,
            acceleration_time_constant=0.02,
            refractory_seconds=0.20,
        )
        settings.update(overrides)
        return HandAccelerationPeakDetector(**settings)

    def test_acceleration_local_maximum_produces_one_peak(self):
        positions = [(0.5, y) for y in (0.50, 0.51, 0.54, 0.59, 0.65, 0.67)]
        self.assertEqual(self._peak_count(self._detector(), positions), 1)

    def test_upward_acceleration_does_not_produce_a_downstroke_peak(self):
        positions = [(0.5, y) for y in (0.67, 0.66, 0.63, 0.58, 0.52, 0.50)]
        self.assertEqual(self._peak_count(self._detector(), positions), 0)

    def test_sideways_acceleration_does_not_produce_a_downstroke_peak(self):
        positions = [(x, 0.5) for x in (0.50, 0.51, 0.54, 0.59, 0.65, 0.67)]
        self.assertEqual(self._peak_count(self._detector(), positions), 0)

    def test_any_direction_option_accepts_an_upward_peak(self):
        positions = [(0.5, y) for y in (0.67, 0.66, 0.63, 0.58, 0.52, 0.50)]
        detector = self._detector(downward_only=False)
        self.assertEqual(self._peak_count(detector, positions), 1)

    def test_downward_only_is_the_interface_default(self):
        self.assertTrue(parse_args([]).conductor_downward_only)
        self.assertFalse(
            parse_args(["--conductor-any-direction"]).conductor_downward_only
        )

    def test_single_melody_hand_moving_up_makes_one_scale_step(self):
        melody = HandMelodyHeight(movement_step=0.03)
        melody.start_beat(0.60)

        self.assertEqual(melody.update(0.565), (1,))
        self.assertEqual(melody.update(0.565), ())

    def test_single_melody_hand_moving_down_descends_scale(self):
        melody = HandMelodyHeight(movement_step=0.03)
        melody.start_beat(0.50)

        self.assertEqual(melody.update(0.54), (-1,))

if __name__ == "__main__":
    unittest.main()
