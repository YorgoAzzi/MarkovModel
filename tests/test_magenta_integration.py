import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from magenta_host import MidiBackend, MockBackend, run_server
from magenta_protocol import (
    INPUT_ALL,
    INPUT_SOPRANO,
    MIDI_ONSET,
    MIDI_SUSTAIN,
    MagentaStatePacket,
)
from magenta_router import MagentaRouter, MagentaStateTracker


def frame(
    *,
    new_voices=("bass", "tenor", "alto", "soprano"),
    released_voices=(),
    sustained_voices=(),
    pitches=None,
):
    pitches = pitches or {"bass": 48, "tenor": 55, "alto": 60, "soprano": 67}
    return SimpleNamespace(
        duration_seconds=0.025,
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


class ProtocolTests(unittest.TestCase):
    def test_packet_round_trip_preserves_128_pitch_state(self):
        notes = [0] * 128
        notes[64] = MIDI_ONSET
        packet = MagentaStatePacket(
            sequence=8,
            sent_at_ns=123,
            enabled=True,
            input_mode=INPUT_SOPRANO,
            notes=tuple(notes),
            velocity=88,
            tempo_bpm=96,
        )

        restored = MagentaStatePacket.from_bytes(packet.to_bytes())

        self.assertEqual(restored, packet)

    def test_packet_rejects_wrong_state_size(self):
        with self.assertRaises(ValueError):
            MagentaStatePacket(0, 0, True, INPUT_ALL, (0,) * 127)


class StateTrackerTests(unittest.TestCase):
    def test_soprano_filter_and_all_voice_filter(self):
        tracker = MagentaStateTracker()
        tracker.consume(frame(), now=1.0)

        soprano = tracker.state(INPUT_SOPRANO)
        self.assertEqual([i for i, value in enumerate(soprano) if value], [67])
        self.assertEqual(soprano[67], MIDI_ONSET)

        all_voices = tracker.state(INPUT_ALL, retrigger=True)
        self.assertEqual(
            [i for i, value in enumerate(all_voices) if value],
            [48, 55, 60, 67],
        )
        self.assertEqual(all_voices[67], MIDI_ONSET)
        self.assertEqual(all_voices[48], MIDI_ONSET)

    def test_release_and_expiration_remove_notes(self):
        tracker = MagentaStateTracker()
        tracker.consume(frame(), now=1.0)
        tracker.state(INPUT_ALL)
        tracker.consume(
            frame(new_voices=(), released_voices=("soprano",)),
            now=1.01,
        )
        state = tracker.state(INPUT_ALL)
        self.assertEqual(state[67], 0)

        tracker.expire(2.0)
        state = tracker.state(INPUT_ALL)
        self.assertFalse(any(state))

    def test_short_onset_is_not_lost_before_one_25_hz_packet(self):
        tracker = MagentaStateTracker()
        tracker.consume(frame(new_voices=("soprano",)), now=1.0)
        tracker.expire(2.0)

        first = tracker.state(INPUT_SOPRANO)
        self.assertEqual(first[67], MIDI_ONSET)
        tracker.expire(2.0)
        self.assertFalse(any(tracker.state(INPUT_SOPRANO)))

    def test_filtered_out_onsets_do_not_prevent_expiration(self):
        tracker = MagentaStateTracker()
        tracker.consume(frame(), now=1.0)
        tracker.state(INPUT_SOPRANO)

        tracker.expire(2.0)

        self.assertFalse(any(tracker.state(INPUT_ALL)))


class FakeMido:
    class Port:
        def __init__(self):
            self.messages = []
            self.closed = False

        def send(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    def __init__(self):
        self.port = self.Port()

    def get_output_names(self):
        return ["Microsoft Synth", "Magenta RT Input 1"]

    def open_output(self, name):
        self.opened = name
        return self.port

    @staticmethod
    def Message(kind, **values):
        return {"type": kind, **values}


class MidiBackendTests(unittest.TestCase):
    def test_midi_backend_forwards_onset_sustain_and_release(self):
        fake = FakeMido()
        backend = MidiBackend(mido_module=fake)
        notes = [0] * 128
        notes[64] = MIDI_ONSET
        packet = MagentaStatePacket(0, 0, True, INPUT_SOPRANO, tuple(notes), 91)
        backend.handle(packet)
        notes[64] = MIDI_SUSTAIN
        backend.handle(MagentaStatePacket(1, 0, True, INPUT_SOPRANO, tuple(notes), 91))
        backend.handle(
            MagentaStatePacket(2, 0, False, INPUT_SOPRANO, (0,) * 128, 91)
        )

        note_messages = fake.port.messages
        self.assertEqual([message["type"] for message in note_messages], ["note_on", "note_off"])
        self.assertEqual(note_messages[0]["velocity"], 91)
        backend.close()
        self.assertTrue(fake.port.closed)


class RouterLoopbackTests(unittest.TestCase):
    def test_router_connects_to_local_mock_gateway(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = MockBackend(log_every=0)
        stop_event = threading.Event()
        server = threading.Thread(
            target=run_server,
            kwargs={"backend": backend, "port": port, "stop_event": stop_event},
            daemon=True,
        )
        server.start()
        router = MagentaRouter(enabled=True, port=port, frame_rate=50).start()
        try:
            router.observe(frame())
            deadline = time.monotonic() + 1.0
            while "Connected" not in router.status() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIn("Connected (mock)", router.status())
            self.assertGreater(backend.frames, 0)
        finally:
            router.close()
            stop_event.set()
            server.join(timeout=1.0)

    def test_disabled_router_ignores_frames_until_reenabled(self):
        router = MagentaRouter(enabled=False)
        router.observe(frame())
        self.assertFalse(router.tracker.active)

        router.set_enabled(True)
        router.observe(frame())
        self.assertEqual(set(router.tracker.active), {"bass", "tenor", "alto", "soprano"})


if __name__ == "__main__":
    unittest.main()
