import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from realtime_performance import MidiPlayer


class FakeMido:
    @staticmethod
    def Message(message_type, **values):
        return {"type": message_type, **values}


class FakePort:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


class MidiExpressionTests(unittest.TestCase):
    def make_player(self):
        player = object.__new__(MidiPlayer)
        player.mido = FakeMido()
        player.port = FakePort()
        player.active = {}
        player.note_timers = {}
        player.lock = threading.Lock()
        return player

    def test_brightness_and_articulation_reach_midi_output(self):
        player = self.make_player()
        frame = SimpleNamespace(
            controls=SimpleNamespace(volume=0.7, brightness=0.8, articulation=0.0),
            new_notes=True,
            planned_duration_ticks=4,
            duration_seconds=0.25,
            pitches={"bass": 48, "tenor": 55, "alto": 60, "soprano": 64},
            velocity=80,
        )

        player.render(frame)
        brightness = [
            message
            for message in player.port.messages
            if message["type"] == "control_change" and message["control"] == 74
        ]
        self.assertEqual(len(brightness), 4)
        self.assertTrue(all(message["value"] == 102 for message in brightness))
        resonance = [
            message
            for message in player.port.messages
            if message["type"] == "control_change" and message["control"] == 71
        ]
        self.assertEqual(len(resonance), 4)
        self.assertTrue(all(message["value"] == 100 for message in resonance))
        self.assertEqual(len(player.note_timers), 4)
        player.close()
        self.assertTrue(player.port.closed)
        self.assertFalse(player.active)

    def test_soprano_onset_does_not_retrigger_accompaniment(self):
        player = self.make_player()
        controls = SimpleNamespace(volume=0.7, brightness=0.5, articulation=0.9)
        first = SimpleNamespace(
            controls=controls,
            new_notes=True,
            new_voices=frozenset({"bass", "tenor", "alto", "soprano"}),
            released_voices=frozenset(),
            voice_duration_ticks={voice: 8 for voice in ("bass", "tenor", "alto", "soprano")},
            planned_duration_ticks=8,
            duration_seconds=0.05,
            pitches={"bass": 48, "tenor": 55, "alto": 60, "soprano": 64},
            velocity=80,
        )
        player.render(first)
        player.port.messages.clear()
        soprano_only = SimpleNamespace(
            **{
                **first.__dict__,
                "new_voices": frozenset({"soprano"}),
                "voice_duration_ticks": {"soprano": 4},
                "pitches": {"bass": 48, "tenor": 55, "alto": 60, "soprano": 67},
            }
        )
        player.render(soprano_only)
        note_messages = [
            message for message in player.port.messages if message["type"].startswith("note_")
        ]
        self.assertEqual({message["channel"] for message in note_messages}, {3})
        player.close()

    def test_sustained_soprano_has_no_timer_and_next_pitch_is_immediate(self):
        player = self.make_player()
        controls = SimpleNamespace(volume=0.7, brightness=0.5, articulation=0.9)
        first = SimpleNamespace(
            controls=controls,
            new_notes=True,
            new_voices=frozenset({"soprano"}),
            released_voices=frozenset(),
            sustained_voices=frozenset({"soprano"}),
            voice_duration_ticks={"soprano": 64},
            planned_duration_ticks=64,
            duration_seconds=0.02,
            pitches={"soprano": 64},
            velocity=80,
        )
        player.render(first)
        self.assertEqual(player.active[3], 64)
        self.assertNotIn(3, player.note_timers)

        moved = SimpleNamespace(**{**first.__dict__, "pitches": {"soprano": 67}})
        player.render(moved)
        note_messages = [
            message for message in player.port.messages if message["type"].startswith("note_")
        ]
        self.assertEqual(note_messages[-2]["type"], "note_off")
        self.assertEqual(note_messages[-2]["note"], 64)
        self.assertEqual(note_messages[-1]["type"], "note_on")
        self.assertEqual(note_messages[-1]["note"], 67)
        player.close()

    def test_stop_all_releases_every_active_voice(self):
        player = self.make_player()
        player.active = {0: 48, 3: 72}
        player.stop_all()
        offs = [message for message in player.port.messages if message["type"] == "note_off"]
        self.assertEqual(len(offs), 2)
        self.assertFalse(player.active)
        player.close()

    def test_timbre_program_change_targets_only_soprano_channel(self):
        player = self.make_player()
        frame = SimpleNamespace(
            controls=SimpleNamespace(volume=0.7, brightness=0.5, articulation=0.9),
            new_notes=False,
            voice_programs={"soprano": 40},
            released_voices=frozenset(),
            pitches={},
        )
        player.render(frame)
        programs = [
            message
            for message in player.port.messages
            if message["type"] == "program_change"
        ]
        self.assertEqual(programs, [{"type": "program_change", "channel": 3, "program": 40}])
        player.close()


if __name__ == "__main__":
    unittest.main()
