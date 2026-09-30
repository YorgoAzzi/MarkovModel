import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from camera_movement import MotionFeatureExtractor, movement_to_music_controls


class MotionFeatureTests(unittest.TestCase):
    def test_motion_produces_bounded_speed_and_acceleration(self):
        features = MotionFeatureExtractor(
            max_speed=1.0,
            max_acceleration=4.0,
            velocity_time_constant=0.01,
            acceleration_time_constant=0.01,
        )
        first = features.update(0.1, 0.5, 0.0)
        moving = features.update(0.2, 0.5, 0.1)
        faster = features.update(0.4, 0.5, 0.2)
        self.assertEqual(first.speed, 0.0)
        self.assertGreater(moving.speed, 0.0)
        self.assertGreater(faster.acceleration, 0.0)
        self.assertLessEqual(faster.speed, 1.0)
        self.assertLessEqual(faster.acceleration, 1.0)

    def test_missing_blob_decays_motion_and_resets_after_dropout(self):
        features = MotionFeatureExtractor(dropout_reset_seconds=0.2)
        features.update(0.1, 0.1, 0.0)
        moving = features.update(0.3, 0.1, 0.1)
        missing = features.missing(0.4)
        self.assertFalse(missing.detected)
        self.assertLess(missing.speed, moving.speed)
        reacquired = features.update(0.9, 0.9, 0.5)
        self.assertEqual(reacquired.speed, 0.0)
        self.assertEqual(reacquired.acceleration, 0.0)

    def test_speed_and_height_control_only_density_and_volume(self):
        fast_and_high = movement_to_music_controls(1.0, 0.0)
        still_and_low = movement_to_music_controls(0.0, 1.0)
        self.assertEqual(fast_and_high, {"density": 1.0, "volume": 1.0})
        self.assertEqual(still_and_low, {"density": 0.0, "volume": 0.0})
        self.assertEqual(set(fast_and_high), {"density", "volume"})


if __name__ == "__main__":
    unittest.main()

