"""Thread-safe soprano event history for the integrated piano roll."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock


@dataclass(frozen=True)
class PianoRollNote:
    """One displayed soprano note on the performance timeline."""

    start_tick: int
    end_tick: int
    pitch: int
    active: bool = False
    ornament: bool = False


class SopranoPianoRoll:
    """Collect only the hand-performed soprano from realtime frames.

    The display timeline advances only while Soprano Performance mode is
    active. This keeps Full SATB passages out of the roll and avoids a large
    empty gap if the performer switches between modes.
    """

    def __init__(
        self,
        *,
        ticks_per_beat: int = 16,
        beats_per_bar: int = 4,
        visible_bars: int = 4,
    ):
        self.ticks_per_beat = max(1, int(ticks_per_beat))
        self.beats_per_bar = max(1, int(beats_per_bar))
        self.ticks_per_bar = self.ticks_per_beat * self.beats_per_bar
        self.visible_bars = max(1, int(visible_bars))
        self._lock = RLock()
        self._notes: list[PianoRollNote] = []
        self._active: PianoRollNote | None = None
        self._active_sustained = False
        self._current_tick = -1
        self._enabled = False
        self._version = 0

    def _finish_active(self, end_tick: int) -> None:
        if self._active is None:
            return
        end_tick = max(self._active.start_tick + 1, int(end_tick))
        self._notes.append(
            replace(self._active, end_tick=end_tick, active=False)
        )
        self._active = None
        self._active_sustained = False

    def observe(self, frame, *, enabled: bool = True) -> None:
        """Consume one generated tick without blocking the audio thread."""
        with self._lock:
            if not enabled:
                if self._enabled:
                    self._finish_active(self._current_tick + 1)
                    self._version += 1
                self._enabled = False
                return

            self._enabled = True
            self._current_tick += 1
            tick = self._current_tick
            released = set(getattr(frame, "released_voices", ()))
            new_voices = set(
                getattr(
                    frame,
                    "new_voices",
                    frame.pitches.keys() if getattr(frame, "new_notes", False) else (),
                )
            )
            sustained = set(getattr(frame, "sustained_voices", ()))

            if "soprano" in released:
                self._finish_active(tick)

            if "soprano" in new_voices and "soprano" in frame.pitches:
                self._finish_active(tick)
                duration = max(
                    1,
                    int(
                        getattr(frame, "voice_duration_ticks", {}).get(
                            "soprano",
                            getattr(frame, "planned_duration_ticks", 1),
                        )
                    ),
                )
                is_sustained = "soprano" in sustained
                self._active = PianoRollNote(
                    start_tick=tick,
                    end_tick=tick + (1 if is_sustained else duration),
                    pitch=max(0, min(127, int(frame.pitches["soprano"]))),
                    active=True,
                    ornament=(not is_sustained and duration <= self.ticks_per_beat // 2),
                )
                self._active_sustained = is_sustained

            if (
                self._active is not None
                and not self._active_sustained
                and tick >= self._active.end_tick
            ):
                self._finish_active(self._active.end_tick)
            self._version += 1

    def snapshot(self) -> dict:
        """Return an immutable snapshot suitable for the Qt UI thread."""
        with self._lock:
            notes = list(self._notes)
            if self._active is not None:
                end_tick = self._active.end_tick
                if self._active_sustained:
                    end_tick = max(end_tick, self._current_tick + 1)
                notes.append(replace(self._active, end_tick=end_tick, active=True))
            measure = max(1, self._current_tick // self.ticks_per_bar + 1)
            if notes:
                status = f"Measure {measure} | {len(notes)} soprano notes"
            else:
                status = "Waiting for performed soprano notes..."
            return {
                "notes": tuple(notes),
                "current_tick": max(0, self._current_tick),
                "ticks_per_beat": self.ticks_per_beat,
                "ticks_per_bar": self.ticks_per_bar,
                "visible_bars": self.visible_bars,
                "version": self._version,
                "status": status,
            }
