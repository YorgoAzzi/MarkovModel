import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from performance_session import MODE_FULL, MODE_SOPRANO, PerformanceSession
from realtime_engine import RealtimeChoraleEngine
from soprano_performance import SopranoGestureController


CHORD = (0, "maj", 4.0)


class SopranoGestureTests(unittest.TestCase):
    def setUp(self):
        self.controller = SopranoGestureController(
            key="C", mode="major", movement_step=0.03
        )

    def observe(
        self,
        y,
        fingers=(True, False, False, False, False),
        t=0.0,
        pinch=False,
        x=None,
        index_extension=None,
    ):
        self.controller.update(
            y=y,
            fingers=fingers,
            pinching=pinch,
            timestamp=t,
            x=x,
            index_extension=index_extension,
        )

    def step(self, tick):
        return self.controller.advance(
            tick_index=tick, ticks_per_beat=16, chord=CHORD
        )

    def test_stationary_hand_holds_one_note_without_retriggering(self):
        self.observe(0.5)
        first = self.step(0)
        held = self.step(1)
        still_held = self.step(64)
        self.assertTrue(first.onset)
        self.assertTrue(first.sustain)
        self.assertFalse(held.onset)
        self.assertTrue(held.sustain)
        self.assertFalse(still_held.onset)
        self.assertEqual(still_held.pitch, first.pitch)
        self.assertEqual(still_held.duration_ticks, 64)

    def test_vertical_motion_steps_immediately_through_scale(self):
        self.observe(0.55, t=0.0)
        first = self.step(0)
        self.observe(0.50, t=0.03)
        upward = self.step(1)
        self.observe(0.56, t=0.06)
        downward = self.step(2)
        self.assertTrue(upward.onset)
        self.assertTrue(upward.sustain)
        self.assertGreater(upward.pitch, first.pitch)
        self.assertTrue(downward.onset)
        self.assertTrue(downward.sustain)
        self.assertLess(downward.pitch, upward.pitch)

    def test_pinch_sustains_and_fist_rests(self):
        self.observe(0.5, t=0.0)
        first = self.step(0)
        self.observe(0.5, fingers=(False,) * 5, t=0.05, pinch=True)
        sustained = self.step(8)
        self.assertEqual(sustained.pitch, first.pitch)
        self.assertFalse(sustained.onset)

        self.observe(0.5, fingers=(False,) * 5, t=0.10)
        self.observe(0.5, fingers=(False,) * 5, t=0.30)
        rested = self.step(9)
        self.assertIsNone(rested.pitch)
        self.assertTrue(rested.released)

    def test_paused_tracking_does_not_queue_a_movement_burst(self):
        self.observe(0.7, t=0.0)
        self.step(0)
        self.controller.synchronize(
            y=0.25,
            fingers=(True, False, False, False, False),
            pinching=False,
            timestamp=1.0,
        )
        resumed = self.step(20)
        following = self.step(21)
        self.assertTrue(resumed.onset)
        self.assertFalse(following.onset)

    def test_horizontal_shake_starts_trill(self):
        for index, x in enumerate((0.50, 0.55, 0.48, 0.56, 0.47)):
            self.observe(0.5, t=index * 0.05, x=x)
        event = self.step(0)
        self.assertTrue(event.onset)
        self.assertEqual(event.gesture, "shake/trill")

    def test_quick_index_bend_and_release_starts_mordent(self):
        self.observe(0.5, t=0.0, index_extension=0.9)
        self.step(0)
        self.observe(0.5, t=0.10, index_extension=0.2)
        self.observe(0.5, t=0.22, index_extension=0.9)
        event = self.step(0)
        self.assertTrue(event.onset)
        self.assertEqual(event.gesture, "index tap/mordent")

    def test_finger_count_selects_soprano_timbre_after_dwell(self):
        three = (False, True, True, True, False)
        self.observe(0.5, fingers=three, t=0.0)
        self.observe(0.5, fingers=three, t=0.25)
        self.assertEqual(self.controller.timbre(), (52, "Choir"))
        four = (False, True, True, True, True)
        self.observe(0.5, fingers=four, t=0.30)
        self.observe(0.5, fingers=four, t=0.55)
        self.assertEqual(self.controller.timbre(), (73, "Flute"))
        five = (True, True, True, True, True)
        self.observe(0.5, fingers=five, t=0.60)
        self.observe(0.5, fingers=five, t=0.85)
        self.assertEqual(self.controller.timbre(), (88, "Synth Pad"))


class SessionIntegrationTests(unittest.TestCase):
    def test_mode_switch_combines_three_voice_model_with_hand_soprano(self):
        engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            ticks_per_beat=16,
            control_smoothing=1.0,
            seed=52,
        )
        soprano = SopranoGestureController()
        session = PerformanceSession(engine, soprano, mode=MODE_SOPRANO)
        soprano.update(
            y=0.5,
            fingers=(True,) * 5,
            pinching=False,
            timestamp=0.0,
        )
        frame = session.generate_tick()
        self.assertEqual(set(frame.pitches), {"bass", "tenor", "alto", "soprano"})
        self.assertIn("soprano", frame.new_voices)
        self.assertIn("soprano", frame.sustained_voices)
        self.assertEqual(frame.voice_duration_ticks["soprano"], 64)
        self.assertEqual(frame.voice_programs, {"soprano": 52})
        self.assertEqual(set(engine.generated_voices), {"bass", "tenor", "alto"})

        session.set_mode(MODE_FULL)
        full = session.generate_tick()
        self.assertEqual(set(full.new_voices), {"bass", "tenor", "alto", "soprano"})

    def test_pause_state_does_not_advance_the_engine(self):
        engine = RealtimeChoraleEngine.from_paths(
            ROOT / "models/factorized_markov_model.npz",
            ROOT / "models/chord_markov_model.npz",
            ROOT / "dataChords",
            ROOT / "cache/progression_blocks_cache.json",
            ticks_per_beat=16,
            seed=53,
        )
        session = PerformanceSession(engine, SopranoGestureController())
        before = engine.tick_index
        session.set_playing(False)
        self.assertFalse(session.is_playing)
        self.assertEqual(engine.tick_index, before)


if __name__ == "__main__":
    unittest.main()
