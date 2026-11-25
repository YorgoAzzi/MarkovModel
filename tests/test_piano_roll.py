import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from piano_roll import SopranoPianoRoll


def frame(
    *,
    pitch=72,
    onset=False,
    released=False,
    sustained=False,
    duration=8,
    tick_index=0,
    duration_seconds=0.05,
):
    return SimpleNamespace(
        pitches={"soprano": pitch},
        new_notes=onset or released,
        new_voices=frozenset({"soprano"} if onset else ()),
        released_voices=frozenset({"soprano"} if released else ()),
        sustained_voices=frozenset({"soprano"} if sustained else ()),
        voice_duration_ticks={"soprano": duration} if onset else {},
        planned_duration_ticks=duration,
        tick_index=tick_index,
        duration_seconds=duration_seconds,
    )


class SopranoPianoRollTests(unittest.TestCase):
    def test_held_note_is_one_continuously_growing_rectangle(self):
        roll = SopranoPianoRoll(ticks_per_beat=16)
        roll.observe(frame(onset=True, sustained=True), enabled=True)
        for _ in range(5):
            roll.observe(frame(sustained=True), enabled=True)

        notes = roll.snapshot()["notes"]

        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].active)
        self.assertEqual((notes[0].start_tick, notes[0].end_tick), (0, 6))

    def test_new_pitch_closes_previous_note_and_release_closes_current(self):
        roll = SopranoPianoRoll(ticks_per_beat=16)
        roll.observe(frame(pitch=72, onset=True, sustained=True), enabled=True)
        roll.observe(frame(pitch=74, onset=True, sustained=True), enabled=True)
        roll.observe(frame(pitch=74, released=True), enabled=True)

        notes = roll.snapshot()["notes"]

        self.assertEqual([(note.pitch, note.start_tick, note.end_tick) for note in notes], [
            (72, 0, 1),
            (74, 1, 2),
        ])
        self.assertTrue(all(not note.active for note in notes))

    def test_full_satb_frames_are_not_added_to_soprano_history(self):
        roll = SopranoPianoRoll(ticks_per_beat=16)
        for _ in range(10):
            roll.observe(frame(onset=True, sustained=True), enabled=False)
        roll.observe(frame(pitch=76, onset=True, sustained=True), enabled=True)

        snapshot = roll.snapshot()

        self.assertEqual(len(snapshot["notes"]), 1)
        self.assertEqual(snapshot["notes"][0].start_tick, 0)

    def test_short_non_sustained_note_is_marked_as_an_ornament(self):
        roll = SopranoPianoRoll(ticks_per_beat=16)
        roll.observe(frame(onset=True, sustained=False, duration=2), enabled=True)

        note = roll.snapshot()["notes"][0]

        self.assertTrue(note.ornament)
        self.assertEqual(note.end_tick, 2)

    def test_conductor_source_ticks_preserve_beat_spacing(self):
        roll = SopranoPianoRoll(ticks_per_beat=16)
        roll.observe(
            frame(onset=True, sustained=True, tick_index=0),
            enabled=True,
            use_source_ticks=True,
        )
        roll.observe(
            frame(sustained=True, tick_index=16),
            enabled=True,
            use_source_ticks=True,
        )

        note = roll.snapshot()["notes"][0]
        self.assertEqual((note.start_tick, note.end_tick), (0, 17))

    def test_conductor_melody_advances_without_another_chord_beat(self):
        roll = SopranoPianoRoll(ticks_per_beat=16)
        roll.observe(frame(pitch=72, onset=True, sustained=True, tick_index=0),
                     use_source_ticks=True, timestamp=10.0)
        roll.observe(frame(pitch=74, onset=True, sustained=True, tick_index=3),
                     use_source_ticks=True, timestamp=10.25)
        roll.observe(frame(pitch=76, onset=True, sustained=True, tick_index=3),
                     use_source_ticks=True, timestamp=11.1)

        snapshot = roll.snapshot(timestamp=11.2)
        self.assertGreater(snapshot["current_tick"], 16)
        self.assertEqual([note.pitch for note in snapshot["notes"]], [72, 74, 76])
        self.assertLess(snapshot["notes"][1].start_tick,
                        snapshot["notes"][2].start_tick)
        self.assertGreater(snapshot["notes"][2].end_tick,
                           snapshot["notes"][2].start_tick)

        paused = roll.snapshot(playing=False, timestamp=11.25)["current_tick"]
        self.assertEqual(roll.snapshot(playing=False, timestamp=12.0)["current_tick"],
                         paused)
        self.assertEqual(roll.snapshot(playing=True, timestamp=12.0)["current_tick"],
                         paused)
        self.assertGreater(roll.snapshot(timestamp=12.2)["current_tick"], paused)


if __name__ == "__main__":
    unittest.main()
