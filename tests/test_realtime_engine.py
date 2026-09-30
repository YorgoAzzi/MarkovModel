import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from realtime_engine import (
    ChordControl,
    MovementBiasedDurationSampler,
    RealtimeChoraleEngine,
    SmoothedControlState,
    chord_token_for_control,
)


class ControlTests(unittest.TestCase):
    def test_controls_are_clamped_smoothed_and_accent_is_one_shot(self):
        controls = SmoothedControlState(smoothing=1.0)
        controls.update(density=2.0, volume=-1.0, tilt=-2.0, rotation=0.5, accent=True)
        first = controls.advance()
        second = controls.advance()
        self.assertEqual(
            (first.density, first.volume, first.tilt, first.rotation),
            (1.0, 0.0, -1.0, 0.5),
        )
        self.assertTrue(first.accent)
        self.assertFalse(second.accent)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            control_smoothing=1.0,
            seed=11,
        )

    def test_rhythm_uses_learned_states_and_chord_state_survives(self):
        engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            control_smoothing=1.0,
            seed=11,
        )
        frames = [engine.generate_tick() for _ in range(16)]
        learned = set(engine.melody_model.duration_model.states)
        sampled = {
            frame.sampled_duration_beats for frame in frames if frame.new_notes
        }
        self.assertTrue(sampled)
        self.assertTrue(sampled.issubset(learned))
        self.assertEqual(len(engine.chord_history), 1)

    def test_every_sonority_stays_in_satb_order(self):
        for _ in range(8):
            pitches = self.engine.generate_tick().pitches
            self.assertLess(pitches["bass"], pitches["tenor"])
            self.assertLess(pitches["tenor"], pitches["alto"])
            self.assertLess(pitches["alto"], pitches["soprano"])

    def test_new_bar_realizes_the_new_chord(self):
        engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            control_smoothing=1.0,
            seed=18,
        )
        frames = [engine.generate_tick() for _ in range(17)]
        self.assertTrue(frames[0].new_notes)
        self.assertTrue(frames[16].new_notes)
        self.assertEqual(len(engine.chord_history), 2)

    def test_accent_waits_for_new_notes(self):
        engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            control_smoothing=1.0,
            seed=12,
        )
        engine.generate_tick()
        engine.update_controls(accent=True)
        frames = [engine.generate_tick() for _ in range(17)]
        attacked = next(frame for frame in frames if frame.new_notes)
        self.assertTrue(attacked.new_notes)
        self.assertTrue(attacked.controls.accent)

    def test_volume_changes_loudness_without_changing_note_speed(self):
        engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            control_smoothing=1.0,
            seed=13,
        )
        first = engine.generate_tick()
        engine.update_controls(volume=0.0)
        following = [engine.generate_tick() for _ in range(17)]
        next_note = next(frame for frame in following if frame.new_notes)
        self.assertLess(next_note.velocity, first.velocity)

    def test_performer_chord_is_applied_at_next_beat(self):
        engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            control_smoothing=1.0,
            seed=19,
        )
        engine.generate_tick()
        engine.update_chord_control(2, "major")
        before_boundary = [engine.generate_tick() for _ in range(3)]
        self.assertTrue(all(frame.chord[0:2] != (2, "maj") for frame in before_boundary))
        boundary = engine.generate_tick()
        self.assertEqual(boundary.tick_in_bar, 4)
        self.assertEqual(boundary.chord[0:2], (2, "maj"))
        self.assertTrue(boundary.new_notes)


class ChordControlTests(unittest.TestCase):
    def test_key_defines_root_and_tilt_policy_defines_quality(self):
        self.assertEqual(
            chord_token_for_control("C", "major", ChordControl(2, "major"))[0:2],
            (2, "maj"),
        )
        self.assertEqual(
            chord_token_for_control("C", "major", ChordControl(2, "minor"))[0:2],
            (2, "min"),
        )
        self.assertEqual(
            chord_token_for_control("C", "major", ChordControl(7, "diatonic"))[0:2],
            (11, "dim"),
        )


class DurationBiasTests(unittest.TestCase):
    class UniformDurationModel:
        states = [0.25, 0.5, 1.0, 2.0, 4.0]
        _state_indexes = {value: index for index, value in enumerate(states)}
        transition_matrix = __import__("numpy").full((5, 5, 5), 0.2)

        @staticmethod
        def _generate_starting_pair():
            return 1.0, 1.0

    def test_high_density_biases_markov_samples_shorter(self):
        import numpy as np

        np.random.seed(21)
        low_sampler = MovementBiasedDurationSampler(self.UniformDurationModel())
        low = [low_sampler.sample(0.0) for _ in range(500)]
        np.random.seed(21)
        high_sampler = MovementBiasedDurationSampler(self.UniformDurationModel())
        high = [high_sampler.sample(1.0) for _ in range(500)]
        self.assertLess(sum(high) / len(high), sum(low) / len(low))
        self.assertTrue(set(high).issubset(set(self.UniformDurationModel.states)))

    def test_neutral_density_preserves_learned_probabilities(self):
        import numpy as np

        sampler = MovementBiasedDurationSampler(self.UniformDurationModel())
        np.testing.assert_allclose(sampler.probabilities(0.5), np.full(5, 0.2))

    def test_truncated_duration_is_corrected_to_a_learned_state(self):
        sampler = MovementBiasedDurationSampler(self.UniformDurationModel())
        corrected = sampler.correct_current(0.6)
        self.assertEqual(corrected, 0.5)
        self.assertEqual(sampler.current, 0.5)


if __name__ == "__main__":
    unittest.main()
