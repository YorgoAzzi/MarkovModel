"""Thread-safe mode and transport layer for the hand performance."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock

try:
    from realtime_engine import ACCOMPANIMENT_VOICES
except ModuleNotFoundError:
    from .realtime_engine import ACCOMPANIMENT_VOICES


MODE_FULL = "full_satb"
MODE_SOPRANO = "soprano"
MODES = (MODE_FULL, MODE_SOPRANO)


class PerformanceSession:
    """Combine the Markov engine, hand soprano, mode selection, and play state."""

    def __init__(self, engine, soprano, *, mode: str = MODE_FULL):
        self.engine = engine
        self.soprano = soprano
        self._lock = RLock()
        self._mode = MODE_FULL
        self._playing = True
        self._revision = 0
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

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def set_playing(self, playing: bool) -> bool:
        with self._lock:
            value = bool(playing)
            if value != self._playing:
                self._playing = value
                self._revision += 1
            return self._playing

    def toggle_playing(self) -> bool:
        with self._lock:
            self._playing = not self._playing
            self._revision += 1
            return self._playing

    def set_mode(self, mode: str) -> str:
        mode = str(mode).strip().lower()
        aliases = {"full": MODE_FULL, "satb": MODE_FULL, "performance": MODE_SOPRANO}
        mode = aliases.get(mode, mode)
        if mode not in MODES:
            raise ValueError(f"Unknown performance mode: {mode}")
        with self._lock:
            self._mode = mode
            self._revision += 1
            self.engine.clear_chord_control(immediate=(mode == MODE_SOPRANO))
            if mode == MODE_SOPRANO:
                self.engine.set_generated_voices(ACCOMPANIMENT_VOICES)
            else:
                self.engine.set_generated_voices(("bass", "tenor", "alto", "soprano"))
            self.soprano.reset()
            return self._mode

    def generate_tick(self):
        with self._lock:
            frame = self.engine.generate_tick()
            if self._mode != MODE_SOPRANO:
                return replace(frame, voice_programs={"soprano": 52})

            step = self.soprano.advance(
                tick_index=frame.tick_index,
                ticks_per_beat=self.engine.ticks_per_beat,
                chord=frame.chord,
            )
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
            if step.sustain and step.pitch is not None:
                sustained_voices.add("soprano")
            durations = dict(frame.voice_duration_ticks)
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
