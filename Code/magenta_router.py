"""Send generated voices directly to Magenta RealTime's virtual MIDI input."""

from __future__ import annotations

from threading import Event, Lock, Thread, Timer


INPUT_SOPRANO = "soprano"
INPUT_ALL = "all"
INPUT_MODES = (INPUT_SOPRANO, INPUT_ALL)
BACKEND_MIDI = "midi"
BACKEND_MOCK = "mock"
BACKENDS = (BACKEND_MIDI, BACKEND_MOCK)
VOICE_ORDER = ("bass", "tenor", "alto", "soprano")
DEFAULT_MAGENTA_PORT = "Magenta RT Input"


class MockMagentaBackend:
    """Record and optionally print MIDI messages without running Magenta."""

    name = BACKEND_MOCK

    def __init__(self, *, log_messages: bool = True):
        self.log_messages = bool(log_messages)
        self.connected = False
        self.error = ""
        self.events: list[dict] = []

    def connect(self) -> bool:
        self.connected = True
        self.error = ""
        return True

    def send(self, kind: str, pitch: int, velocity: int = 0) -> bool:
        if not self.connected:
            return False
        event = {
            "type": str(kind),
            "note": int(pitch),
            "velocity": int(velocity),
        }
        self.events.append(event)
        if self.log_messages:
            print(
                f"Magenta mock: {event['type']} note={event['note']} "
                f"velocity={event['velocity']}"
            )
        return True

    def close(self) -> None:
        self.connected = False

    @property
    def description(self) -> str:
        return "mock"


class MidiMagentaBackend:
    """Open and write directly to the official MRT2 virtual MIDI port."""

    name = BACKEND_MIDI

    def __init__(self, port_name: str = DEFAULT_MAGENTA_PORT, *, mido_module=None):
        self.requested_port = str(port_name)
        self.mido = mido_module
        self.port = None
        self.port_name = ""
        self.connected = False
        self.error = ""

    def connect(self) -> bool:
        if self.connected and self.port is not None:
            return True
        try:
            if self.mido is None:
                import mido

                self.mido = mido
            names = self.mido.get_output_names()
            matches = [
                name for name in names if self.requested_port.lower() in name.lower()
            ]
            if not matches:
                self.error = f"MIDI port not found: {self.requested_port}"
                return False
            self.port_name = matches[0]
            self.port = self.mido.open_output(self.port_name)
            self.connected = True
            self.error = ""
            print(f"Magenta MIDI output: {self.port_name}")
            return True
        except Exception as exc:
            self.error = f"MIDI connection error: {exc}"
            self.connected = False
            self.port = None
            return False

    def send(self, kind: str, pitch: int, velocity: int = 0) -> bool:
        if not self.connected or self.port is None:
            return False
        try:
            self.port.send(
                self.mido.Message(
                    kind,
                    channel=0,
                    note=max(0, min(127, int(pitch))),
                    velocity=max(0, min(127, int(velocity))),
                )
            )
            return True
        except Exception as exc:
            self.error = f"MIDI send error: {exc}"
            self._disconnect()
            return False

    def _disconnect(self) -> None:
        port = self.port
        self.port = None
        self.connected = False
        if port is not None:
            try:
                port.close()
            except Exception:
                pass

    def close(self) -> None:
        self._disconnect()

    @property
    def description(self) -> str:
        return self.port_name or self.requested_port


class MagentaRouter:
    """Filter generated voices and forward their MIDI events directly to MRT2."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        input_mode: str = INPUT_SOPRANO,
        backend: str = BACKEND_MIDI,
        port_name: str = DEFAULT_MAGENTA_PORT,
        reconnect_seconds: float = 1.0,
        log_mock: bool = True,
        backend_instance=None,
    ):
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown Magenta input mode: {input_mode}")
        if backend not in BACKENDS:
            raise ValueError(f"Unknown Magenta backend: {backend}")
        if backend_instance is not None:
            self.backend = backend_instance
        elif backend == BACKEND_MOCK:
            self.backend = MockMagentaBackend(log_messages=log_mock)
        else:
            self.backend = MidiMagentaBackend(port_name)
        self.backend_name = backend
        self.reconnect_seconds = max(0.1, float(reconnect_seconds))
        self._enabled = bool(enabled)
        self._playing = True
        self._input_mode = input_mode
        self._active: dict[str, int] = {}
        self._timers: dict[str, Timer] = {}
        self._bootstrap_next = bool(enabled)
        self._lock = Lock()
        self._stop = Event()
        self._wake = Event()
        self._connector: Thread | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def input_mode(self) -> str:
        with self._lock:
            return self._input_mode

    @property
    def active(self) -> dict[str, int]:
        with self._lock:
            return dict(self._active)

    def start(self) -> "MagentaRouter":
        if self._connector is not None and self._connector.is_alive():
            return self
        self._stop.clear()
        self._wake.clear()
        self._connector = Thread(
            target=self._connection_loop,
            name="magenta-midi-connector",
            daemon=True,
        )
        self._connector.start()
        self._wake.set()
        return self

    def _connection_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                should_connect = self._enabled and not self.backend.connected
            if should_connect and self.backend.connect():
                with self._lock:
                    self._bootstrap_next = True
            self._wake.wait(self.reconnect_seconds)
            self._wake.clear()

    def _selected_voices(self) -> tuple[str, ...]:
        return ("soprano",) if self._input_mode == INPUT_SOPRANO else VOICE_ORDER

    def _send_locked(self, kind: str, pitch: int, velocity: int = 0) -> bool:
        sent = self.backend.send(kind, pitch, velocity)
        if not sent:
            self._bootstrap_next = True
            self._wake.set()
        return sent

    def _cancel_timer_locked(self, voice: str) -> None:
        timer = self._timers.pop(voice, None)
        if timer is not None:
            timer.cancel()

    def _release_voice_locked(self, voice: str, expected_pitch: int | None = None) -> None:
        pitch = self._active.get(voice)
        if pitch is None or (expected_pitch is not None and pitch != expected_pitch):
            return
        self._cancel_timer_locked(voice)
        self._active.pop(voice, None)
        if pitch not in self._active.values():
            self._send_locked("note_off", pitch, 0)

    def _release_timer(self, voice: str, pitch: int) -> None:
        with self._lock:
            self._release_voice_locked(voice, expected_pitch=pitch)

    def _start_voice_locked(
        self,
        voice: str,
        pitch: int,
        velocity: int,
        gate_seconds: float | None,
    ) -> None:
        self._release_voice_locked(voice)
        pitch = max(0, min(127, int(pitch)))
        already_sounding = pitch in self._active.values()
        self._active[voice] = pitch
        if not already_sounding:
            self._send_locked("note_on", pitch, velocity)
        if gate_seconds is not None:
            timer = Timer(
                max(0.025, float(gate_seconds)),
                self._release_timer,
                args=(voice, pitch),
            )
            timer.daemon = True
            self._timers[voice] = timer
            timer.start()

    def _stop_all_locked(self) -> None:
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        pitches = sorted(set(self._active.values()))
        self._active.clear()
        for pitch in pitches:
            self._send_locked("note_off", pitch, 0)

    def observe(self, frame) -> None:
        with self._lock:
            if not self._enabled or not self._playing:
                return
            if not self.backend.connected:
                self._bootstrap_next = True
                self._wake.set()
                return

            selected = set(self._selected_voices())
            released = set(getattr(frame, "released_voices", ()))
            sustained = set(getattr(frame, "sustained_voices", ()))
            new_voices = set(
                getattr(
                    frame,
                    "new_voices",
                    frame.pitches.keys() if getattr(frame, "new_notes", False) else (),
                )
            )
            durations = getattr(frame, "voice_duration_ticks", {})
            articulation = max(
                0.0,
                min(1.0, float(frame.controls.articulation)),
            )
            gate_ratio = 0.20 + 0.78 * articulation
            velocity = max(1, min(127, int(getattr(frame, "velocity", 90))))

            if self._bootstrap_next:
                self._stop_all_locked()
                for voice in self._selected_voices():
                    if voice in released or voice not in frame.pitches:
                        continue
                    gate_seconds = None
                    if voice in new_voices and voice not in sustained:
                        duration_ticks = max(
                            1,
                            int(
                                durations.get(
                                    voice,
                                    getattr(frame, "planned_duration_ticks", 1),
                                )
                            ),
                        )
                        gate_seconds = (
                            duration_ticks * float(frame.duration_seconds) * gate_ratio
                        )
                    self._start_voice_locked(
                        voice,
                        frame.pitches[voice],
                        velocity,
                        gate_seconds,
                    )
                self._bootstrap_next = False
                return

            for voice in released:
                if voice in self._active:
                    self._release_voice_locked(voice)
            for voice in sustained:
                if voice in selected:
                    self._cancel_timer_locked(voice)
            for voice in new_voices:
                if voice not in selected or voice not in frame.pitches:
                    continue
                duration_ticks = max(
                    1,
                    int(
                        durations.get(
                            voice,
                            getattr(frame, "planned_duration_ticks", 1),
                        )
                    ),
                )
                gate_seconds = None
                if voice not in sustained:
                    gate_seconds = (
                        duration_ticks * float(frame.duration_seconds) * gate_ratio
                    )
                self._start_voice_locked(
                    voice,
                    frame.pitches[voice],
                    velocity,
                    gate_seconds,
                )

    def set_enabled(self, enabled: bool) -> bool:
        with self._lock:
            value = bool(enabled)
            if value != self._enabled:
                if not value:
                    self._stop_all_locked()
                self._enabled = value
                self._bootstrap_next = value
                self._wake.set()
            return self._enabled

    def set_input_mode(self, input_mode: str) -> str:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown Magenta input mode: {input_mode}")
        with self._lock:
            if input_mode != self._input_mode:
                self._stop_all_locked()
                self._input_mode = input_mode
                self._bootstrap_next = self._enabled and self._playing
            return self._input_mode

    def set_playing(self, playing: bool) -> bool:
        with self._lock:
            value = bool(playing)
            if value != self._playing:
                if not value:
                    self._stop_all_locked()
                self._playing = value
                self._bootstrap_next = value and self._enabled
            return self._playing

    def reset_output(self) -> None:
        with self._lock:
            self._stop_all_locked()
            self._bootstrap_next = self._enabled and self._playing

    def status(self) -> str:
        with self._lock:
            if not self._enabled:
                return "Off"
            if self.backend.connected:
                return f"Connected ({self.backend.description})"
            return self.backend.error or "Connecting to Magenta MIDI input..."

    def close(self) -> None:
        with self._lock:
            self._stop_all_locked()
            self._enabled = False
        self._stop.set()
        self._wake.set()
        if self._connector is not None:
            self._connector.join(timeout=2.0)
        self.backend.close()
