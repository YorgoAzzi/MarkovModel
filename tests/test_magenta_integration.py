import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from magenta_router import (
    INPUT_ALL,
    INPUT_SOPRANO,
    MagentaRouter,
    MidiMagentaBackend,
    MockMagentaBackend,
)


VOICES = ("bass", "tenor", "alto", "soprano")


def frame(
    *,
    new_voices=VOICES,
    released_voices=(),
    sustained_voices=VOICES,
    pitches=None,
    duration_seconds=0.025,
):
    pitches = pitches or {"bass": 48, "tenor": 55, "alto": 60, "soprano": 67}
    return SimpleNamespace(
        duration_seconds=duration_seconds,
        pitches=pitches,
        controls=SimpleNamespace(articulation=1.0),
        velocity=84,
        new_notes=bool(new_voices or released_voices),
        new_voices=frozenset(new_voices),
        released_voices=frozenset(released_voices),
        sustained_voices=frozenset(sustained_voices),
        voice_duration_ticks={voice: 4 for voice in new_voices},
        planned_duration_ticks=4,
    )


class FakeMido:
    class Port:
        def __init__(self):
            self.messages = []
            self.closed = False

        def send(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    def __init__(self, names=None):
        self.names = names or ["Microsoft Synth", "Magenta RT Input 1"]
        self.port = self.Port()
        self.opened = None

    def get_output_names(self):
        return list(self.names)

    def open_output(self, name):
        self.opened = name
        return self.port

    @staticmethod
    def Message(kind, **values):
        return {"type": kind, **values}


class MidiBackendTests(unittest.TestCase):
    def test_backend_finds_the_magenta_virtual_port_and_sends_direct_midi(self):
        fake = FakeMido()
        backend = MidiMagentaBackend(mido_module=fake)

        self.assertTrue(backend.connect())
        self.assertEqual(fake.opened, "Magenta RT Input 1")
        backend.send("note_on", 64, 91)
        backend.send("note_off", 64, 0)

        self.assertEqual(
            [message["type"] for message in fake.port.messages],
            ["note_on", "note_off"],
        )
        self.assertEqual(fake.port.messages[0]["velocity"], 91)
        backend.close()
        self.assertTrue(fake.port.closed)

    def test_missing_virtual_port_is_reported_without_crashing(self):
        backend = MidiMagentaBackend(mido_module=FakeMido(["Microsoft Synth"]))

        self.assertFalse(backend.connect())
        self.assertIn("MIDI port not found", backend.error)


class DirectRouterTests(unittest.TestCase):
    def make_router(self, *, input_mode=INPUT_SOPRANO, enabled=True):
        backend = MockMagentaBackend(log_messages=False)
        backend.connect()
        router = MagentaRouter(
            enabled=enabled,
            input_mode=input_mode,
            backend="mock",
            backend_instance=backend,
        )
        return router, backend

    def test_soprano_filter_sends_only_the_final_soprano(self):
        router, backend = self.make_router(input_mode=INPUT_SOPRANO)

        router.observe(frame())

        self.assertEqual(
            backend.events,
            [{"type": "note_on", "note": 67, "velocity": 84}],
        )
        self.assertEqual(router.active, {"soprano": 67})
        router.close()

    def test_all_filter_sends_all_four_voices(self):
        router, backend = self.make_router(input_mode=INPUT_ALL)

        router.observe(frame())

        note_ons = [event for event in backend.events if event["type"] == "note_on"]
        self.assertEqual({event["note"] for event in note_ons}, {48, 55, 60, 67})
        self.assertEqual(set(router.active), set(VOICES))
        router.close()

    def test_input_change_releases_old_notes_and_bootstraps_new_filter(self):
        router, backend = self.make_router(input_mode=INPUT_SOPRANO)
        current = frame()
        router.observe(current)

        router.set_input_mode(INPUT_ALL)
        router.observe(current)

        self.assertIn({"type": "note_off", "note": 67, "velocity": 0}, backend.events)
        self.assertEqual(set(router.active), set(VOICES))
        router.close()

    def test_pause_disable_and_close_release_active_notes(self):
        router, backend = self.make_router(input_mode=INPUT_ALL)
        router.observe(frame())

        router.set_playing(False)

        note_offs = [event for event in backend.events if event["type"] == "note_off"]
        self.assertEqual({event["note"] for event in note_offs}, {48, 55, 60, 67})
        self.assertFalse(router.active)
        router.close()

    def test_non_sustained_note_uses_articulation_gate(self):
        router, backend = self.make_router(input_mode=INPUT_SOPRANO)
        router.observe(
            frame(
                new_voices=("soprano",),
                sustained_voices=(),
                duration_seconds=0.01,
            )
        )

        deadline = time.monotonic() + 0.3
        while router.active and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(router.active)
        self.assertEqual(backend.events[-1]["type"], "note_off")
        router.close()

    def test_disabled_router_ignores_frames_until_enabled(self):
        router, backend = self.make_router(enabled=False)
        router.observe(frame())
        self.assertFalse(backend.events)

        router.set_enabled(True)
        router.observe(frame())

        self.assertEqual(backend.events[0]["type"], "note_on")
        router.close()

    def test_router_status_uses_backend_connection_state(self):
        router, _backend = self.make_router()
        self.assertEqual(router.status(), "Connected (mock)")
        router.set_enabled(False)
        self.assertEqual(router.status(), "Off")
        router.close()


if __name__ == "__main__":
    unittest.main()
