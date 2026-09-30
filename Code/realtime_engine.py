"""Low-latency, stateful movement-to-SATB generation.

This module deliberately contains no input loop and no audio clock.  A phone,
GUI, OSC/WebSocket receiver, or test program can update controls at any time;
the caller asks for one clock tick whenever its scheduler is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

try:
    import markovchain as core
except ModuleNotFoundError:  # Also support `from Code.realtime_engine import ...`.
    from . import markovchain as core


@dataclass(frozen=True)
class MovementControls:
    """Normalized movement values consumed by the generator."""

    density: float = 0.0  # 0..1: longer to shorter generated notes
    volume: float = 0.7  # 0..1: quiet to loud MIDI dynamics
    tilt: float = 0.0  # -1..1: low to high register
    rotation: float = 0.0  # 0..1: narrow to wide melodic motion
    accent: bool = False  # one-shot velocity accent


@dataclass(frozen=True)
class GeneratedTick:
    tick_index: int
    bar_index: int
    tick_in_bar: int
    duration_seconds: float
    pitches: dict[str, int]
    chord: tuple[int, str, float]
    controls: MovementControls
    new_notes: bool
    velocity: int
    sampled_duration_beats: float
    planned_duration_ticks: int


@dataclass(frozen=True)
class ChordControl:
    """A performer-selected scale degree and chord-quality policy."""

    degree: int
    quality: str = "diatonic"  # minor, diatonic, or major


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def chord_token_for_control(
    key: str,
    mode: str,
    control: ChordControl,
    duration_beats: float = 4.0,
) -> tuple[int, str, float]:
    """Build a triad while keeping the key for roots but allowing quality override."""
    degree = int(control.degree)
    if degree not in range(1, 8):
        raise ValueError("Chord degree must be between 1 and 7")
    quality_policy = str(control.quality).strip().lower()
    if quality_policy not in {"minor", "diatonic", "major"}:
        raise ValueError("Chord quality must be minor, diatonic, or major")

    intervals = list(core._mode_scale_intervals(mode))
    index = degree - 1
    root_offset = intervals[index]
    root_pc = (core._parse_key_root(key) + root_offset) % 12
    if quality_policy == "major":
        quality = "maj"
    elif quality_policy == "minor":
        quality = "min"
    else:
        third = (intervals[(index + 2) % 7] - root_offset) % 12
        fifth = (intervals[(index + 4) % 7] - root_offset) % 12
        quality = {
            (4, 7): "maj",
            (3, 7): "min",
            (3, 6): "dim",
        }.get((third, fifth))
        if quality is None:
            raise ValueError(
                f"Unsupported diatonic triad on degree {degree}: ({third}, {fifth})"
            )
    return int(root_pc), quality, float(duration_beats)


class SmoothedControlState:
    """Thread-safe target values with an accent latch and tick-rate smoothing."""

    def __init__(self, smoothing: float = 0.35):
        self.smoothing = _clamp(smoothing, 0.0, 1.0)
        self._target = MovementControls()
        self._current = MovementControls()
        self._accent_pending = False
        self._lock = Lock()

    def update(
        self,
        *,
        density: Optional[float] = None,
        volume: Optional[float] = None,
        tilt: Optional[float] = None,
        rotation: Optional[float] = None,
        accent: bool = False,
    ) -> None:
        with self._lock:
            self._target = MovementControls(
                density=self._target.density if density is None else _clamp(density, 0.0, 1.0),
                volume=self._target.volume if volume is None else _clamp(volume, 0.0, 1.0),
                tilt=self._target.tilt if tilt is None else _clamp(tilt, -1.0, 1.0),
                rotation=self._target.rotation if rotation is None else _clamp(rotation, 0.0, 1.0),
            )
            self._accent_pending = self._accent_pending or bool(accent)

    def advance(self) -> MovementControls:
        with self._lock:
            alpha = self.smoothing
            current = self._current
            target = self._target
            self._current = MovementControls(
                density=current.density + alpha * (target.density - current.density),
                volume=current.volume + alpha * (target.volume - current.volume),
                tilt=current.tilt + alpha * (target.tilt - current.tilt),
                rotation=current.rotation + alpha * (target.rotation - current.rotation),
                accent=self._accent_pending,
            )
            self._accent_pending = False
            return self._current

    def target(self) -> MovementControls:
        with self._lock:
            return self._target


class MovementBiasedDurationSampler:
    """Samples learned second-order rhythms with a movement-dependent bias."""

    def __init__(self, duration_model, bias_strength: float = 3.5):
        self.model = duration_model
        self.bias_strength = max(0.0, float(bias_strength))
        first, second = self.model._generate_starting_pair()
        self.previous = first
        self.current = second

        durations = np.asarray([float(value) for value in self.model.states], dtype=float)
        if np.any(durations <= 0):
            raise ValueError("Duration model states must all be positive")
        log_durations = np.log(durations)
        spread = float(log_durations.max() - log_durations.min())
        if spread <= 1e-12:
            self.shortness = np.zeros_like(log_durations)
        else:
            position = (log_durations - log_durations.min()) / spread
            self.shortness = 1.0 - 2.0 * position

    def probabilities(self, density: float) -> np.ndarray:
        i = self.model._state_indexes[self.previous]
        j = self.model._state_indexes[self.current]
        learned = np.asarray(self.model.transition_matrix[i, j], dtype=float).copy()
        direction = 2.0 * _clamp(density, 0.0, 1.0) - 1.0
        weights = np.exp(self.bias_strength * direction * self.shortness)
        adjusted = learned * weights
        total = float(adjusted.sum())
        if total <= 0 or not np.isfinite(total):
            adjusted = np.ones(len(self.model.states), dtype=float)
            total = float(adjusted.sum())
        return adjusted / total

    def sample(self, density: float) -> float:
        probabilities = self.probabilities(density)
        index = int(np.random.choice(len(self.model.states), p=probabilities))
        next_duration = self.model.states[index]
        self.previous, self.current = self.current, next_duration
        return float(next_duration)

    def correct_current(self, performed_duration: float) -> float:
        """Keep Markov history aligned when a movement truncates a long note."""
        closest = min(
            self.model.states,
            key=lambda duration: abs(float(duration) - float(performed_duration)),
        )
        self.current = closest
        return float(closest)


class PersistentProgression:
    """Samples learned four-chord blocks without regenerating prior bars."""

    def __init__(
        self,
        chord_model,
        progression_blocks_by_mode,
        laplace_alpha: float = 1.0,
        cadence_every_bars: int = 4,
    ):
        self.chord_model = chord_model
        self.blocks = progression_blocks_by_mode
        self.laplace_alpha = float(laplace_alpha)
        self.cadence_every_bars = max(0, int(cadence_every_bars))
        self._family = None
        self._model = None
        self._symbols: list[str] = []
        self._block_pair = None
        self._bootstrap_blocks = []

    def _reset_for_family(self, family: str) -> None:
        available = list(self.blocks.get(family) or [])
        if not available:
            raise ValueError(f"No learned progression blocks are available for {family} mode.")
        self._family = family
        self._model = core._build_progression_block_model(
            family, laplace_alpha=self.laplace_alpha, progression_blocks=available
        )
        first, second = self._model._generate_starting_pair()
        self._block_pair = (first, second)
        self._bootstrap_blocks = [first, second]
        self._symbols = []

    def _next_symbol(self, family: str) -> str:
        if family != self._family:
            self._reset_for_family(family)
        if not self._symbols:
            if self._bootstrap_blocks:
                block = self._bootstrap_blocks.pop(0)
            else:
                previous, current = self._block_pair
                next_block = self._model._generate_next(previous, current)
                self._block_pair = (current, next_block)
                block = next_block
            self._symbols.extend(str(symbol) for symbol in block)
        return self._symbols.pop(0)

    def next_chord(
        self,
        *,
        bar_index: int,
        key_root_pc: int,
        mode: str,
        scale_pitch_classes: set[int],
        tension: float,
    ) -> tuple[int, str, float]:
        family = core._mode_family(mode)
        symbol = self._next_symbol(family)

        # Tension delays cadences by up to roughly half the base phrase length.
        cadence = self.cadence_every_bars
        if cadence > 1:
            cadence += int(round(_clamp(tension, 0.0, 1.0) * cadence * 0.5))
            phrase_position = int(bar_index) % cadence
            if phrase_position == cadence - 2:
                symbol = "V"
            elif phrase_position == cadence - 1:
                symbol = "i" if family == "minor" else "I"

        return core._roman_symbol_to_chord_token(
            symbol,
            chord_model=self.chord_model,
            key_root_pc=key_root_pc,
            mode=mode,
            scale_pitch_classes=scale_pitch_classes,
        )


class RealtimeChoraleEngine:
    """Persistent SATB engine advanced on a fixed, short musical clock."""

    def __init__(
        self,
        melody_model,
        chord_model,
        progression_blocks_by_mode,
        *,
        tempo_bpm: float = 120.0,
        beats_per_bar: int = 4,
        ticks_per_beat: int = 4,
        key: str = "C",
        mode: str = "major",
        beam_width: int = 20,
        candidates_per_voice: int = 6,
        top_sonorities: int = 24,
        repeat_note_penalty: float = 5.0,
        cadence_every_bars: int = 4,
        control_smoothing: float = 0.35,
        duration_bias_strength: float = 3.5,
        seed: Optional[int] = None,
    ):
        if tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be positive")
        if beats_per_bar < 1 or ticks_per_beat < 1:
            raise ValueError("beats_per_bar and ticks_per_beat must be positive")
        if seed is not None:
            np.random.seed(int(seed))

        self.melody_model = melody_model
        self.chord_model = chord_model
        self.tempo_bpm = float(tempo_bpm)
        self.beats_per_bar = int(beats_per_bar)
        self.ticks_per_beat = int(ticks_per_beat)
        self.ticks_per_bar = self.beats_per_bar * self.ticks_per_beat
        self.key = str(key)
        self.mode = core._canonical_mode(mode)
        self.beam_width = max(1, int(beam_width))
        self.candidates_per_voice = max(1, int(candidates_per_voice))
        self.top_sonorities = max(1, int(top_sonorities))
        self.repeat_note_penalty = max(0.0, float(repeat_note_penalty))
        self.controls = SmoothedControlState(control_smoothing)
        self.progression = PersistentProgression(
            chord_model,
            progression_blocks_by_mode,
            cadence_every_bars=cadence_every_bars,
        )
        self.interval_pairs = core._init_interval_pairs(melody_model)
        self.duration_sampler = MovementBiasedDurationSampler(
            melody_model.duration_model,
            bias_strength=duration_bias_strength,
        )
        self.beams = [core._default_chorale_beam()]
        self.tick_index = 0
        self.note_index = 0
        self.current_chord = None
        self.current_sonority = None
        self.ticks_since_note = 0
        self.sampled_duration_beats = 1.0
        self.planned_duration_ticks = self.ticks_per_beat
        self.chord_history: list[tuple[int, str, float]] = []
        self._chord_control_lock = Lock()
        self._requested_chord_control: Optional[ChordControl] = None
        self._applied_chord_control: Optional[ChordControl] = None

    @classmethod
    def from_paths(
        cls,
        melody_model_path="models/factorized_markov_model.npz",
        chord_model_path="models/chord_markov_model.npz",
        chord_data_folder="dataChords",
        progression_cache_path="cache/progression_blocks_cache.json",
        **kwargs,
    ):
        melody_model = core.FactorizedMelodyModel.load(Path(melody_model_path))
        chord_model = core.ChordMarkovModel.load(Path(chord_model_path))
        blocks = core.load_progression_blocks_from_folder(
            chord_data_folder, progression_cache_path, refresh_cache=False
        )
        return cls(melody_model, chord_model, blocks, **kwargs)

    @property
    def tick_seconds(self) -> float:
        return 60.0 / self.tempo_bpm / self.ticks_per_beat

    def update_controls(self, **values) -> None:
        self.controls.update(**values)

    def update_chord_control(self, degree: int, quality: str = "diatonic") -> None:
        """Request a performer chord; it is applied at the next beat boundary."""
        control = ChordControl(int(degree), str(quality).strip().lower())
        # Validate now so camera-thread errors do not surface in the audio thread.
        chord_token_for_control(self.key, self.mode, control, self.beats_per_bar)
        with self._chord_control_lock:
            self._requested_chord_control = control

    def clear_chord_control(self) -> None:
        """Return harmony to the learned progression at the next bar boundary."""
        with self._chord_control_lock:
            self._requested_chord_control = None

    def chord_control_target(self) -> Optional[ChordControl]:
        with self._chord_control_lock:
            return self._requested_chord_control

    def configure(self, *, key: Optional[str] = None, mode: Optional[str] = None) -> None:
        if key is not None:
            core._parse_key_root(key)
            self.key = str(key)
        if mode is not None:
            self.mode = core._canonical_mode(mode)
        if key is not None or mode is not None:
            # Rebuild a held hand-selected degree in the new scale.
            self._applied_chord_control = None

    def _maximum_duration_ticks(self, density: float) -> int:
        """Bound response time while retaining sampled Markov durations."""
        whole_note_ticks = self.beats_per_bar * self.ticks_per_beat
        density = _clamp(density, 0.0, 1.0)
        return max(1, int(round(1 + (1.0 - density) * (whole_note_ticks - 1))))

    def _sample_note_duration(self, density: float) -> None:
        self.sampled_duration_beats = self.duration_sampler.sample(density)
        self.planned_duration_ticks = max(
            1,
            int(round(self.sampled_duration_beats * self.ticks_per_beat)),
        )

    def _generate_sonority(self, controls: MovementControls) -> dict[str, int]:
        interval_deltas = {}
        if self.note_index > 0:
            for voice in core.VOICE_ORDER:
                delta = core._consume_interval_delta(
                    self.melody_model, self.interval_pairs, voice
                )
                activity = 0.55 + 1.45 * controls.rotation
                interval_deltas[voice] = int(round(delta * activity))

        chord_classes = core._chord_token_to_pitch_classes(
            int(self.current_chord[0]), str(self.current_chord[1])
        )
        centers = {
            voice: (bounds[0] + bounds[1]) // 2
            for voice, bounds in core.VOICE_RANGES.items()
        }
        register_scale = {"bass": 5, "tenor": 7, "alto": 9, "soprano": 12}
        expanded = []

        for beam in self.beams:
            previous = beam["last"]
            targets = dict(beam["targets"])
            for voice in core.VOICE_ORDER:
                moving_target = targets[voice] + interval_deltas.get(voice, 0)
                register_target = centers[voice] + round(register_scale[voice] * controls.tilt)
                targets[voice] = int(round(0.72 * moving_target + 0.28 * register_target))

            candidates = {}
            for voice in core.VOICE_ORDER:
                previous_pitch = None if previous is None else int(previous[voice])
                candidates[voice] = core._build_ranked_voice_candidates(
                    chord_classes,
                    voice,
                    targets[voice],
                    previous_pitch,
                    self.candidates_per_voice,
                )

            repeat_penalty = self.repeat_note_penalty * (1.0 - 0.6 * controls.rotation)
            choices = core._enumerate_sonority_candidates(
                candidates,
                previous,
                targets,
                self.top_sonorities,
                repeat_streaks=beam["repeat_streaks"],
                repeated_note_base_penalty=repeat_penalty,
            )
            for sonority, local_cost in choices:
                streaks = {}
                for voice in core.VOICE_ORDER:
                    repeated = previous is not None and sonority[voice] == previous[voice]
                    streaks[voice] = beam["repeat_streaks"][voice] + 1 if repeated else 0
                expanded.append(
                    {
                        "score": float(beam["score"] + local_cost),
                        "last": sonority,
                        "targets": dict(sonority),
                        "repeat_streaks": streaks,
                    }
                )

        if not expanded:
            raise RuntimeError("Realtime generation could not find a valid SATB sonority")
        expanded.sort(key=lambda item: item["score"])
        best_score = expanded[0]["score"]
        # Removing the shared accumulated cost avoids unbounded numbers in long sessions.
        self.beams = expanded[: self.beam_width]
        for beam in self.beams:
            beam["score"] -= best_score
        self.note_index += 1
        return dict(self.beams[0]["last"])

    def generate_tick(self) -> GeneratedTick:
        controls = self.controls.advance()
        tick_in_bar = self.tick_index % self.ticks_per_bar
        bar_index = self.tick_index // self.ticks_per_bar
        beat_boundary = tick_in_bar % self.ticks_per_beat == 0
        with self._chord_control_lock:
            requested_chord = self._requested_chord_control
        apply_performer_chord = (
            requested_chord is not None
            and beat_boundary
            and (
                self.current_chord is None
                or requested_chord != self._applied_chord_control
            )
        )
        advance_learned_chord = requested_chord is None and (
            tick_in_bar == 0 or self.current_chord is None
        )
        chord_boundary = False
        if apply_performer_chord:
            chord_boundary = self.current_chord is not None
            self.current_chord = chord_token_for_control(
                self.key,
                self.mode,
                requested_chord,
                self.beats_per_bar,
            )
            self._applied_chord_control = requested_chord
            self.chord_history.append(self.current_chord)
        elif advance_learned_chord:
            chord_boundary = self.current_chord is not None
            scale = core._build_scale_pitch_classes(self.key, self.mode)
            self.current_chord = self.progression.next_chord(
                bar_index=bar_index,
                key_root_pc=core._parse_key_root(self.key),
                mode=self.mode,
                scale_pitch_classes=scale,
                tension=controls.rotation,
            )
            self._applied_chord_control = None
            self.chord_history.append(self.current_chord)

        if self.current_sonority is None:
            new_notes = True
        else:
            self.ticks_since_note += 1
            effective_duration = min(
                self.planned_duration_ticks,
                self._maximum_duration_ticks(controls.density),
            )
            new_notes = chord_boundary or self.ticks_since_note >= effective_duration
        if new_notes:
            if (
                self.current_sonority is not None
                and self.ticks_since_note < self.planned_duration_ticks
            ):
                performed_beats = self.ticks_since_note / self.ticks_per_beat
                self.duration_sampler.correct_current(performed_beats)
            self._sample_note_duration(controls.density)
            self.current_sonority = self._generate_sonority(controls)
            self.ticks_since_note = 0
        elif controls.accent:
            # Keep a gesture accent until there is an audible note onset.
            self.controls.update(accent=True)

        velocity = int(round(25 + 92 * controls.volume + (16 if controls.accent else 0)))
        result = GeneratedTick(
            tick_index=self.tick_index,
            bar_index=bar_index,
            tick_in_bar=tick_in_bar,
            duration_seconds=self.tick_seconds,
            pitches=dict(self.current_sonority),
            chord=self.current_chord,
            controls=controls,
            new_notes=new_notes,
            velocity=max(1, min(127, velocity)),
            sampled_duration_beats=self.sampled_duration_beats,
            planned_duration_ticks=self.planned_duration_ticks,
        )
        self.tick_index += 1
        return result
