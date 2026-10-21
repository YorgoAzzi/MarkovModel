"""Convert generated SATB frames to Magenta RealTime MIDI conditioning."""

from __future__ import annotations

from dataclasses import dataclass
import socket
from threading import Event, Lock, Thread
import time

try:
    from magenta_protocol import (
        DEFAULT_FRAME_RATE,
        DEFAULT_HOST,
        DEFAULT_PORT,
        INPUT_ALL,
        INPUT_MODES,
        INPUT_SOPRANO,
        MIDI_ONSET,
        MIDI_SUSTAIN,
        MagentaStatePacket,
        parse_ack,
    )
except ModuleNotFoundError:
    from .magenta_protocol import (
        DEFAULT_FRAME_RATE,
        DEFAULT_HOST,
        DEFAULT_PORT,
        INPUT_ALL,
        INPUT_MODES,
        INPUT_SOPRANO,
        MIDI_ONSET,
        MIDI_SUSTAIN,
        MagentaStatePacket,
        parse_ack,
    )


VOICE_ORDER = ("bass", "tenor", "alto", "soprano")


@dataclass
class ActiveConditioningNote:
    pitch: int
    velocity: int
    off_at: float | None


class MagentaStateTracker:
    """Track the sounding note state independently from the MIDI player."""

    def __init__(self, ticks_per_beat: int = 16):
        self.ticks_per_beat = max(1, int(ticks_per_beat))
        self.active: dict[str, ActiveConditioningNote] = {}
        self.pending_onsets: set[int] = set()
        self.latest_tempo = 120.0
        self.latest_velocity = 90

    def clear(self) -> None:
        self.active.clear()
        self.pending_onsets.clear()

    def consume(self, frame, now: float, *, bootstrap: bool = False) -> None:
        self.latest_velocity = max(1, min(127, int(getattr(frame, "velocity", 90))))
        duration_seconds = max(1e-6, float(frame.duration_seconds))
        self.latest_tempo = 60.0 / (duration_seconds * self.ticks_per_beat)
        released = set(getattr(frame, "released_voices", ()))
        sustained = set(getattr(frame, "sustained_voices", ()))
        new_voices = set(
            getattr(
                frame,
                "new_voices",
                frame.pitches.keys() if getattr(frame, "new_notes", False) else (),
            )
        )
        if bootstrap:
            new_voices.update(
                voice for voice in VOICE_ORDER if voice in getattr(frame, "pitches", {})
            )

        for voice in released:
            self.active.pop(voice, None)
        for voice in sustained:
            note = self.active.get(voice)
            if note is not None:
                note.off_at = None

        durations = getattr(frame, "voice_duration_ticks", {})
        articulation = max(0.0, min(1.0, float(frame.controls.articulation)))
        gate_ratio = 0.20 + 0.78 * articulation
        for voice in new_voices:
            if voice not in VOICE_ORDER or voice not in frame.pitches:
                continue
            pitch = max(0, min(127, int(frame.pitches[voice])))
            old = self.active.get(voice)
            if old is not None and old.pitch != pitch:
                self.pending_onsets.discard(old.pitch)
            duration_ticks = max(
                1,
                int(durations.get(voice, getattr(frame, "planned_duration_ticks", 1))),
            )
            off_at = None
            if voice not in sustained and not bootstrap:
                off_at = now + max(0.025, duration_ticks * duration_seconds * gate_ratio)
            self.active[voice] = ActiveConditioningNote(
                pitch=pitch,
                velocity=self.latest_velocity,
                off_at=off_at,
            )
            self.pending_onsets.add(pitch)

    def expire(self, now: float) -> None:
        expired = [
            voice
            for voice, note in self.active.items()
            if note.off_at is not None
            and note.off_at <= now
            and note.pitch not in self.pending_onsets
        ]
        for voice in expired:
            self.active.pop(voice, None)

    @staticmethod
    def _selected_voices(input_mode: str):
        return ("soprano",) if input_mode == INPUT_SOPRANO else VOICE_ORDER

    def state(self, input_mode: str, *, retrigger: bool = False) -> tuple[int, ...]:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown Magenta input mode: {input_mode}")
        notes = [0] * 128
        selected = self._selected_voices(input_mode)
        selected_pitches = {
            self.active[voice].pitch for voice in selected if voice in self.active
        }
        for pitch in selected_pitches:
            notes[pitch] = (
                MIDI_ONSET
                if retrigger or pitch in self.pending_onsets
                else MIDI_SUSTAIN
            )
        # One conditioning frame consumes all pending edges. Notes excluded by
        # the current filter are retriggered explicitly if the filter changes.
        self.pending_onsets.clear()
        return tuple(notes)


class MagentaRouter:
    """Publish conditioning frames to a gateway on this computer at 25 Hz."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        input_mode: str = INPUT_SOPRANO,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        frame_rate: float = DEFAULT_FRAME_RATE,
        ticks_per_beat: int = 16,
        clock=time.perf_counter,
    ):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("The Magenta gateway is local-only; use 127.0.0.1")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown Magenta input mode: {input_mode}")
        self.host = DEFAULT_HOST
        self.port = int(port)
        self.frame_rate = max(1.0, float(frame_rate))
        self.clock = clock
        self.tracker = MagentaStateTracker(ticks_per_beat=ticks_per_beat)
        self._enabled = bool(enabled)
        self._playing = True
        self._input_mode = input_mode
        self._sequence = 0
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._socket: socket.socket | None = None
        self._last_ack_at = 0.0
        self._backend = ""
        self._backend_status = ""
        self._pending_silence = True
        self._retrigger = bool(enabled)
        self._bootstrap_next = bool(enabled)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def input_mode(self) -> str:
        with self._lock:
            return self._input_mode

    def start(self) -> "MagentaRouter":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((DEFAULT_HOST, 0))
        self._socket.setblocking(False)
        self._thread = Thread(target=self._run, name="magenta-router", daemon=True)
        self._thread.start()
        return self

    def observe(self, frame) -> None:
        now = self.clock()
        with self._lock:
            if not self._enabled or not self._playing:
                return
            bootstrap = self._bootstrap_next and self._enabled and self._playing
            self.tracker.consume(frame, now, bootstrap=bootstrap)
            if bootstrap:
                self._bootstrap_next = False
                self._retrigger = True

    def set_enabled(self, enabled: bool) -> bool:
        with self._lock:
            value = bool(enabled)
            if value != self._enabled:
                self._enabled = value
                self.tracker.clear()
                self._pending_silence = True
                self._retrigger = value
                self._bootstrap_next = value
            return self._enabled

    def set_input_mode(self, input_mode: str) -> str:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown Magenta input mode: {input_mode}")
        with self._lock:
            if input_mode != self._input_mode:
                self._input_mode = input_mode
                self._pending_silence = True
                self._retrigger = True
            return self._input_mode

    def set_playing(self, playing: bool) -> bool:
        with self._lock:
            value = bool(playing)
            if value != self._playing:
                self._playing = value
                self._pending_silence = True
                self._bootstrap_next = value
                self._retrigger = value
                if not value:
                    self.tracker.clear()
            return self._playing

    def reset_output(self) -> None:
        with self._lock:
            self.tracker.clear()
            self._pending_silence = True
            self._bootstrap_next = self._enabled and self._playing
            self._retrigger = self._bootstrap_next

    def status(self) -> str:
        with self._lock:
            if not self._enabled:
                return "Off"
            age = self.clock() - self._last_ack_at
            if self._last_ack_at > 0.0 and age < 2.0:
                detail = self._backend or "local gateway"
                if self._backend_status and self._backend_status != "ready":
                    detail = f"{detail}: {self._backend_status}"
                return f"Connected ({detail})"
            return f"Waiting for local gateway on 127.0.0.1:{self.port}"

    def _packet(self, now: float) -> MagentaStatePacket:
        with self._lock:
            self.tracker.expire(now)
            enabled = self._enabled and self._playing
            if self._pending_silence:
                notes = (0,) * 128
                self._pending_silence = False
            elif enabled:
                notes = self.tracker.state(
                    self._input_mode,
                    retrigger=self._retrigger,
                )
                self._retrigger = False
            else:
                notes = (0,) * 128
            packet = MagentaStatePacket(
                sequence=self._sequence,
                sent_at_ns=time.time_ns(),
                enabled=enabled,
                input_mode=self._input_mode,
                notes=notes,
                velocity=self.tracker.latest_velocity,
                tempo_bpm=self.tracker.latest_tempo,
            )
            self._sequence += 1
            return packet

    def _receive_acks(self, now: float) -> None:
        if self._socket is None:
            return
        while True:
            try:
                payload, _address = self._socket.recvfrom(2048)
            except BlockingIOError:
                return
            except OSError:
                return
            try:
                ack = parse_ack(payload)
            except (ValueError, KeyError, TypeError):
                continue
            with self._lock:
                self._last_ack_at = now
                self._backend = str(ack.get("backend", "local gateway"))
                self._backend_status = str(ack.get("status", "ready"))

    def _run(self) -> None:
        interval = 1.0 / self.frame_rate
        deadline = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            packet = self._packet(now)
            try:
                if self._socket is not None:
                    self._socket.sendto(packet.to_bytes(), (self.host, self.port))
                    self._receive_acks(now)
            except OSError:
                pass
            deadline += interval
            remaining = deadline - self.clock()
            if remaining > 0:
                self._stop.wait(remaining)
            else:
                deadline = self.clock()

    def close(self) -> None:
        self.set_enabled(False)
        if self._socket is not None:
            try:
                self._socket.sendto(
                    self._packet(self.clock()).to_bytes(),
                    (self.host, self.port),
                )
            except OSError:
                pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
