"""Low-latency hand-performed soprano layered over generated accompaniment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Sequence

try:
    import markovchain as core
except ModuleNotFoundError:
    from . import markovchain as core


# Zero-based General MIDI programs selected by the total extended-finger count.
SOPRANO_TIMBRES = {
    1: (0, "Piano"),
    2: (40, "Violin"),
    3: (52, "Choir"),
    4: (73, "Flute"),
    5: (88, "Synth Pad"),
}


@dataclass(frozen=True)
class SopranoStep:
    pitch: Optional[int]
    onset: bool
    released: bool
    duration_ticks: int
    gesture: str
    sustain: bool = False


class SopranoGestureController:
    """Translate right-hand motion, taps, and poses into soprano events."""

    def __init__(
        self,
        *,
        key: str = "C",
        mode: str = "major",
        movement_step: float = 0.032,
        fist_dwell_seconds: float = 0.16,
        timbre_dwell_seconds: float = 0.22,
        shake_speed: float = 0.35,
        shake_window_seconds: float = 0.65,
    ):
        self.lock = Lock()
        self.movement_step = max(0.008, float(movement_step))
        self.fist_dwell_seconds = max(0.0, float(fist_dwell_seconds))
        self.timbre_dwell_seconds = max(0.0, float(timbre_dwell_seconds))
        self.shake_speed = max(0.05, float(shake_speed))
        self.shake_window_seconds = max(0.25, float(shake_window_seconds))
        self.key = str(key)
        self.mode = core._canonical_mode(mode)
        self.scale = self._scale_pitches()
        self.scale_index = min(range(len(self.scale)), key=lambda i: abs(self.scale[i] - 72))
        self.current_program = 52
        self.current_timbre = "Choir"
        self._reset_motion_state()

    def _reset_motion_state(self) -> None:
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.last_motion_timestamp: Optional[float] = None
        self.vertical_remainder = 0.0
        self.pending_scale_steps = 0
        self.fingers = (False,) * 5
        self.pinching = False
        self.fist_since: Optional[float] = None
        self.resting = False
        self.active_pitch: Optional[int] = None
        self.force_onset = True
        self.last_onset_tick = -10_000
        self.horizontal_sign = 0
        self.shake_reversals: deque[float] = deque(maxlen=8)
        self.trill_active = False
        self.trill_expires = 0.0
        self.trill_high = False
        self.index_was_extended: Optional[bool] = None
        self.index_bent_at: Optional[float] = None
        self.mordent: deque[int] = deque()
        self.timbre_candidate: Optional[int] = None
        self.timbre_candidate_since = 0.0
        self.last_timestamp = 0.0
        self.gesture = "ready"

    def _scale_pitches(self) -> list[int]:
        pcs = set(core._build_scale_pitch_classes(self.key, self.mode))
        return [pitch for pitch in range(60, 85) if pitch % 12 in pcs]

    def configure(self, *, key: Optional[str] = None, mode: Optional[str] = None) -> None:
        with self.lock:
            current = self.scale[self.scale_index]
            if key is not None:
                core._parse_key_root(key)
                self.key = str(key)
            if mode is not None:
                self.mode = core._canonical_mode(mode)
            self.scale = self._scale_pitches()
            self.scale_index = min(
                range(len(self.scale)), key=lambda index: abs(self.scale[index] - current)
            )
            self.force_onset = True

    def reset(self) -> None:
        with self.lock:
            self.current_program = 52
            self.current_timbre = "Choir"
            self._reset_motion_state()

    def _update_timbre(self, timestamp: float) -> None:
        count = sum(self.fingers)
        if self.pinching or count not in SOPRANO_TIMBRES:
            self.timbre_candidate = None
            return
        if count != self.timbre_candidate:
            self.timbre_candidate = count
            self.timbre_candidate_since = timestamp
            return
        if timestamp - self.timbre_candidate_since >= self.timbre_dwell_seconds:
            self.current_program, self.current_timbre = SOPRANO_TIMBRES[count]

    def _update_shake(self, x: float, timestamp: float) -> None:
        if self.last_x is not None and self.last_motion_timestamp is not None:
            dt = max(1e-4, timestamp - self.last_motion_timestamp)
            dx = x - self.last_x
            speed = abs(dx) / dt
            if speed >= self.shake_speed and abs(dx) >= 0.008:
                sign = 1 if dx > 0 else -1
                if self.horizontal_sign and sign != self.horizontal_sign:
                    self.shake_reversals.append(timestamp)
                self.horizontal_sign = sign
        self.last_x = x
        self.last_motion_timestamp = timestamp
        cutoff = timestamp - self.shake_window_seconds
        while self.shake_reversals and self.shake_reversals[0] < cutoff:
            self.shake_reversals.popleft()
        if len(self.shake_reversals) >= 3:
            self.trill_active = True
            self.trill_expires = timestamp + 0.32
            self.gesture = "shake/trill"
            self.shake_reversals.popleft()
        elif timestamp > self.trill_expires:
            self.trill_active = False

    def _update_index_tap(self, amount: Optional[float], timestamp: float) -> None:
        if amount is None or self.pinching:
            return
        extended = float(amount) >= 0.68
        bent = float(amount) <= 0.42
        if self.index_was_extended is None:
            self.index_was_extended = extended
            return
        if self.index_was_extended and bent:
            self.index_was_extended = False
            self.index_bent_at = timestamp
            return
        if (
            not self.index_was_extended
            and extended
            and self.index_bent_at is not None
            and timestamp - self.index_bent_at <= 0.35
        ):
            self.mordent = deque((0, 1, 0))
            self.force_onset = True
            self.gesture = "index tap/mordent"
            self.index_bent_at = None
            self.index_was_extended = True
        elif self.index_bent_at is not None and timestamp - self.index_bent_at > 0.35:
            self.index_bent_at = None
            self.index_was_extended = extended

    def update(
        self,
        *,
        y: float,
        fingers: Sequence[bool],
        pinching: bool,
        timestamp: float,
        x: Optional[float] = None,
        index_extension: Optional[float] = None,
    ) -> None:
        if len(fingers) != 5:
            raise ValueError("Expected five finger states")
        with self.lock:
            timestamp = float(timestamp)
            y = max(0.0, min(1.0, float(y)))
            if self.last_y is None:
                target = round((1.0 - y) * (len(self.scale) - 1))
                self.scale_index = max(0, min(len(self.scale) - 1, target))
                self.force_onset = True
            else:
                self.vertical_remainder += self.last_y - y
                while self.vertical_remainder >= self.movement_step:
                    self.pending_scale_steps += 1
                    self.vertical_remainder -= self.movement_step
                while self.vertical_remainder <= -self.movement_step:
                    self.pending_scale_steps -= 1
                    self.vertical_remainder += self.movement_step
                self.pending_scale_steps = max(-8, min(8, self.pending_scale_steps))
            self.last_y = y
            self.fingers = tuple(bool(value) for value in fingers)
            self.pinching = bool(pinching)
            self.last_timestamp = timestamp
            if x is not None:
                self._update_shake(max(0.0, min(1.0, float(x))), timestamp)
            self._update_index_tap(index_extension, timestamp)
            self._update_timbre(timestamp)

            is_fist = not any(self.fingers) and not self.pinching
            if is_fist:
                if self.fist_since is None:
                    self.fist_since = timestamp
                elif timestamp - self.fist_since >= self.fist_dwell_seconds:
                    self.resting = True
                    self.gesture = "fist/rest"
            else:
                if self.resting:
                    self.force_onset = True
                self.resting = False
                self.fist_since = None
            if self.pinching:
                self.gesture = "pinch/sustain"

    def missing(self, timestamp: float) -> None:
        with self.lock:
            if float(timestamp) - self.last_timestamp > 0.35:
                self.last_x = None
                self.last_y = None
                self.last_motion_timestamp = None
                self.vertical_remainder = 0.0
                self.pending_scale_steps = 0
                self.trill_active = False

    def synchronize(
        self,
        *,
        y: float,
        fingers: Sequence[bool],
        pinching: bool,
        timestamp: float,
        x: Optional[float] = None,
        index_extension: Optional[float] = None,
    ) -> None:
        """Track the hand while paused without queuing notes or ornaments."""
        if len(fingers) != 5:
            raise ValueError("Expected five finger states")
        with self.lock:
            y = max(0.0, min(1.0, float(y)))
            self.scale_index = max(
                0,
                min(len(self.scale) - 1, round((1.0 - y) * (len(self.scale) - 1))),
            )
            self.last_x = None if x is None else float(x)
            self.last_y = y
            self.last_motion_timestamp = float(timestamp)
            self.vertical_remainder = 0.0
            self.pending_scale_steps = 0
            self.fingers = tuple(bool(value) for value in fingers)
            self.pinching = bool(pinching)
            self.resting = not any(self.fingers) and not self.pinching
            self.fist_since = float(timestamp) if self.resting else None
            self.active_pitch = None
            self.force_onset = not self.resting
            self.horizontal_sign = 0
            self.shake_reversals.clear()
            self.trill_active = False
            self.trill_expires = 0.0
            self.mordent.clear()
            self.index_was_extended = (
                None if index_extension is None else float(index_extension) >= 0.68
            )
            self.index_bent_at = None
            self.last_timestamp = float(timestamp)
            count = sum(self.fingers)
            if count in SOPRANO_TIMBRES and not self.pinching:
                self.current_program, self.current_timbre = SOPRANO_TIMBRES[count]
            self.gesture = "paused"

    def advance(
        self,
        *,
        tick_index: int,
        ticks_per_beat: int,
        chord: tuple[int, str, float],
    ) -> SopranoStep:
        del chord
        with self.lock:
            sixteenth = max(1, int(ticks_per_beat) // 4)
            ornament = max(1, int(ticks_per_beat) // 8)
            held_note = max(1, int(ticks_per_beat) * 4)
            if self.resting:
                released = self.active_pitch is not None
                self.active_pitch = None
                return SopranoStep(None, False, released, sixteenth, self.gesture)
            if self.pinching and self.active_pitch is not None:
                return SopranoStep(
                    self.active_pitch,
                    False,
                    False,
                    held_note,
                    self.gesture,
                    sustain=True,
                )

            if self.trill_active:
                if int(tick_index) - self.last_onset_tick >= ornament or self.force_onset:
                    base = self.scale[self.scale_index]
                    neighbor = self.scale[min(len(self.scale) - 1, self.scale_index + 1)]
                    self.trill_high = not self.trill_high
                    self.active_pitch = neighbor if self.trill_high else base
                    self.last_onset_tick = int(tick_index)
                    self.force_onset = False
                    return SopranoStep(self.active_pitch, True, False, ornament, "shake/trill")
                return SopranoStep(self.active_pitch, False, False, ornament, "shake/trill")

            if self.mordent and (
                int(tick_index) - self.last_onset_tick >= ornament or self.force_onset
            ):
                offset = self.mordent.popleft()
                pitch_index = min(len(self.scale) - 1, self.scale_index + offset)
                self.active_pitch = self.scale[pitch_index]
                self.last_onset_tick = int(tick_index)
                self.force_onset = False
                return SopranoStep(
                    self.active_pitch, True, False, ornament, "index tap/mordent"
                )

            if self.pending_scale_steps:
                direction = 1 if self.pending_scale_steps > 0 else -1
                self.pending_scale_steps -= direction
                self.scale_index = max(
                    0, min(len(self.scale) - 1, self.scale_index + direction)
                )
                self.active_pitch = self.scale[self.scale_index]
                self.last_onset_tick = int(tick_index)
                self.force_onset = False
                self.gesture = "scale up" if direction > 0 else "scale down"
                return SopranoStep(
                    self.active_pitch,
                    True,
                    False,
                    held_note,
                    self.gesture,
                    sustain=True,
                )

            if self.force_onset or self.active_pitch is None:
                self.active_pitch = self.scale[self.scale_index]
                self.last_onset_tick = int(tick_index)
                self.force_onset = False
                self.gesture = "holding"
                return SopranoStep(
                    self.active_pitch,
                    True,
                    False,
                    held_note,
                    self.gesture,
                    sustain=True,
                )
            self.gesture = "holding"
            return SopranoStep(
                self.active_pitch,
                False,
                False,
                held_note,
                self.gesture,
                sustain=True,
            )

    def status(self) -> tuple[str, Optional[int]]:
        with self.lock:
            return self.gesture, self.active_pitch

    def timbre(self) -> tuple[int, str]:
        with self.lock:
            return self.current_program, self.current_timbre
