"""Thread-safe mode and transport layer for the hand performance."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from threading import Condition, RLock

try:
    from realtime_engine import ACCOMPANIMENT_VOICES
except ModuleNotFoundError:
    from .realtime_engine import ACCOMPANIMENT_VOICES


MODE_FULL = "full_satb"
MODE_SOPRANO = "soprano"
MODE_CONDUCTOR = "conductor"
MODES = (MODE_FULL, MODE_SOPRANO, MODE_CONDUCTOR)


class PerformanceSession:
    """Combine the Markov engine, hand soprano, mode selection, and play state."""

    def __init__(self, engine, soprano, *, mode: str = MODE_FULL):
        self.engine = engine
        self.soprano = soprano
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._mode = MODE_FULL
        self._playing = True
        self._revision = 0
        self._conductor_events = deque()
        self._conductor_last_frame = None
        self._conductor_beat_tick = 0
        self._conductor_last_event_tick = 0
        self._conductor_beat_timestamp = 0.0
        self.set_mode(mode)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def tick_seconds(self):
        return self.engine.tick_seconds

    @property
    def event_driven(self) -> bool:
        with self._lock:
            return self._mode == MODE_CONDUCTOR

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def set_playing(self, playing: bool) -> bool:
        with self._lock:
            value = bool(playing)
            if value != self._playing:
                self._playing = value
                self._revision += 1
                if not value:
                    self._conductor_events.clear()
                self._condition.notify_all()
            return self._playing

    def toggle_playing(self) -> bool:
        with self._lock:
            self._playing = not self._playing
            self._revision += 1
            if not self._playing:
                self._conductor_events.clear()
            self._condition.notify_all()
            return self._playing

    def set_mode(self, mode: str) -> str:
        mode = str(mode).strip().lower()
        aliases = {
            "full": MODE_FULL,
            "satb": MODE_FULL,
            "performance": MODE_SOPRANO,
            "conduct": MODE_CONDUCTOR,
        }
        mode = aliases.get(mode, mode)
        if mode not in MODES:
            raise ValueError(f"Unknown performance mode: {mode}")
        with self._lock:
            self._mode = mode
            self._revision += 1
            self._conductor_events.clear()
            self._conductor_last_frame = None
            self.engine.clear_chord_control(
                immediate=(mode in {MODE_SOPRANO, MODE_CONDUCTOR})
            )
            if mode in {MODE_SOPRANO, MODE_CONDUCTOR}:
                self.engine.set_generated_voices(ACCOMPANIMENT_VOICES)
            else:
                self.engine.set_generated_voices(("bass", "tenor", "alto", "soprano"))
            self.soprano.reset()
            self._condition.notify_all()
            return self._mode

    def request_conductor_beat(
        self,
        *,
        degree=None,
        strength: float = 0.0,
        simultaneous: bool = False,
        tempo_bpm=None,
        timestamp: float = 0.0,
    ) -> bool:
        if degree is not None and not 1 <= int(degree) <= 7:
            raise ValueError("Conductor chord degree must be between 1 and 7")
        with self._condition:
            if self._mode != MODE_CONDUCTOR or not self._playing:
                return False
            self._conductor_events.append(
                {
                    "kind": "beat",
                    "degree": None if degree is None else int(degree),
                    "strength": max(0.0, min(1.0, float(strength))),
                    "simultaneous": bool(simultaneous),
                    "tempo_bpm": None if tempo_bpm is None else float(tempo_bpm),
                    "timestamp": float(timestamp),
                }
            )
            self._condition.notify()
            return True

    def request_conductor_melody_step(
        self,
        direction: int,
        *,
        timestamp: float = 0.0,
    ) -> bool:
        direction = 1 if int(direction) > 0 else -1
        with self._condition:
            if (
                self._mode != MODE_CONDUCTOR
                or not self._playing
                or (
                    self._conductor_last_frame is None
                    and not any(
                        event.get("kind") == "beat"
                        for event in self._conductor_events
                    )
                )
            ):
                return False
            self._conductor_events.append(
                {
                    "kind": "melody",
                    "direction": direction,
                    "timestamp": float(timestamp),
                }
            )
            self._condition.notify()
            return True

    def wait_for_frame(self, stop_event, timeout: float = 0.05):
        """Wait for a hand command in Conductor mode; generate normally otherwise."""
        with self._condition:
            if self._mode != MODE_CONDUCTOR:
                return self.generate_tick()
            if not self._playing:
                self._condition.wait(timeout=max(0.001, float(timeout)))
                return None
            if not self._conductor_events and not stop_event.is_set():
                self._condition.wait(timeout=max(0.001, float(timeout)))
            if (
                stop_event.is_set()
                or self._mode != MODE_CONDUCTOR
                or not self._playing
                or not self._conductor_events
            ):
                return None
            event = self._conductor_events.popleft()
            return self._generate_conductor_event(event)

    def _apply_soprano_step(self, frame, step, *, sustain_accompaniment=False):
        pitches = dict(frame.pitches)
        if step.pitch is not None:
            pitches["soprano"] = int(step.pitch)
        new_voices = set(frame.new_voices)
        if step.onset:
            new_voices.add("soprano")
        released_voices = set(frame.released_voices)
        if step.released:
            released_voices.add("soprano")
        sustained_voices = set(frame.sustained_voices)
        if sustain_accompaniment:
            sustained_voices.update(ACCOMPANIMENT_VOICES)
        if step.sustain and step.pitch is not None:
            sustained_voices.add("soprano")
        durations = dict(frame.voice_duration_ticks)
        if sustain_accompaniment:
            durations.update(
                {voice: self.engine.ticks_per_beat for voice in ACCOMPANIMENT_VOICES}
            )
        if step.onset:
            durations["soprano"] = int(step.duration_ticks)
        program, _name = self.soprano.timbre()
        return replace(
            frame,
            pitches=pitches,
            new_notes=bool(new_voices or released_voices),
            new_voices=frozenset(new_voices),
            released_voices=frozenset(released_voices),
            sustained_voices=frozenset(sustained_voices),
            voice_duration_ticks=durations,
            voice_programs={"soprano": program} if step.onset else {},
            planned_duration_ticks=(
                min(durations.values()) if durations else frame.planned_duration_ticks
            ),
        )

    def _generate_conductor_event(self, event):
        if event["kind"] == "beat":
            if event["degree"] is not None:
                self.engine.update_chord_control(event["degree"], "diatonic")
            if event["tempo_bpm"] is not None:
                self.engine.update_tempo(event["tempo_bpm"])
            if event["simultaneous"] or event["strength"] >= 0.70:
                self.engine.update_controls(accent=True)
            frame = self.engine.generate_conducted_beat()
            step = self.soprano.advance(
                tick_index=frame.tick_index,
                ticks_per_beat=self.engine.ticks_per_beat,
                chord=frame.chord,
            )
            frame = self._apply_soprano_step(
                frame,
                step,
                sustain_accompaniment=True,
            )
            self._conductor_last_frame = frame
            self._conductor_beat_tick = frame.tick_index
            self._conductor_last_event_tick = frame.tick_index
            self._conductor_beat_timestamp = event["timestamp"]
            return frame

        if self._conductor_last_frame is None:
            return None
        self.soprano.queue_scale_steps(event["direction"])
        elapsed = max(0.0, event["timestamp"] - self._conductor_beat_timestamp)
        offset = max(
            1,
            min(
                self.engine.ticks_per_beat - 1,
                int(round(elapsed / max(1e-6, self.engine.tick_seconds))),
            ),
        )
        event_tick = max(
            self._conductor_last_event_tick,
            self._conductor_beat_tick + offset,
        )
        base = self._conductor_last_frame
        step = self.soprano.advance(
            tick_index=event_tick,
            ticks_per_beat=self.engine.ticks_per_beat,
            chord=base.chord,
        )
        empty_frame = replace(
            base,
            tick_index=event_tick,
            bar_index=event_tick // self.engine.ticks_per_bar,
            tick_in_bar=event_tick % self.engine.ticks_per_bar,
            new_notes=False,
            new_voices=frozenset(),
            released_voices=frozenset(),
            sustained_voices=frozenset(ACCOMPANIMENT_VOICES),
            voice_duration_ticks={},
            voice_programs={},
        )
        frame = self._apply_soprano_step(
            empty_frame,
            step,
            sustain_accompaniment=True,
        )
        self._conductor_last_frame = frame
        self._conductor_last_event_tick = event_tick
        return frame

    def generate_tick(self):
        with self._lock:
            if self._mode == MODE_CONDUCTOR:
                return None
            frame = self.engine.generate_tick()
            if self._mode != MODE_SOPRANO:
                return replace(frame, voice_programs={"soprano": 52})

            step = self.soprano.advance(
                tick_index=frame.tick_index,
                ticks_per_beat=self.engine.ticks_per_beat,
                chord=frame.chord,
            )
            return self._apply_soprano_step(frame, step)
