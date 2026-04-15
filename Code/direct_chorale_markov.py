import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from music21 import converter, metadata, note, stream

SUPPORTED_EXTENSIONS = {".mid", ".midi", ".xml", ".musicxml", ".mxl"}
VOICE_ORDER = ["bass", "tenor", "alto", "soprano"]
VOICE_RANGES = {
    "bass": (36, 60),
    "tenor": (45, 67),
    "alto": (52, 76),
    "soprano": (60, 84),
}
DEFAULT_DATA_FOLDER = "dataMelody"
DEFAULT_CACHE_PATH = "cache/direct_chorale_states_cache.json"
DEFAULT_MODEL_PATH = "models/direct_chorale_markov.npz"
DEFAULT_GRID_STEP = 1.0
DEFAULT_LENGTH = 32


def _load_cache(cache_path):
    path = Path(cache_path)
    if not path.exists():
        return {"files": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data
    except Exception:
        pass
    return {"files": {}}


def _save_cache(cache_path, cache_data):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)


def _transpose_score_to_reference_key(score):
    try:
        analyzed_key = score.analyze("key")
    except Exception:
        return score
    from music21 import interval as m21interval, pitch as m21pitch

    target_tonic = m21pitch.Pitch("C") if analyzed_key.mode == "major" else m21pitch.Pitch("A")
    transposition = m21interval.Interval(analyzed_key.tonic, target_tonic)
    return score.transpose(transposition)


def _choose_satb_parts(score):
    parts = [part for part in score.parts if len(part.flatten().notes) > 0]
    if len(parts) < 4:
        return None

    scored_parts = []
    for part in parts:
        midi_values = [int(n.pitch.midi) for n in part.flatten().notes if isinstance(n, note.Note)]
        if not midi_values:
            continue
        avg_pitch = float(sum(midi_values)) / len(midi_values)
        scored_parts.append((avg_pitch, part))
    if len(scored_parts) < 4:
        return None

    scored_parts.sort(key=lambda pair: pair[0])
    selected = [pair[1] for pair in scored_parts[:4]]
    return {
        "bass": selected[0].flatten(),
        "tenor": selected[1].flatten(),
        "alto": selected[2].flatten(),
        "soprano": selected[3].flatten(),
    }


def _active_pitch_at_offset(flat_part, offset):
    for el in flat_part.notes:
        if not isinstance(el, note.Note):
            continue
        start = float(el.offset)
        end = start + float(el.duration.quarterLength)
        if start <= offset < end:
            return int(el.pitch.midi)
    return None


def _extract_state_sequence_from_score(score, grid_step=DEFAULT_GRID_STEP):
    normalized_score = _transpose_score_to_reference_key(score)
    satb_parts = _choose_satb_parts(normalized_score)
    if satb_parts is None:
        return []

    max_end = 0.0
    for voice in VOICE_ORDER:
        for el in satb_parts[voice].notes:
            if not isinstance(el, note.Note):
                continue
            max_end = max(max_end, float(el.offset) + float(el.duration.quarterLength))

    total_steps = max(1, int(np.ceil(max_end / float(grid_step))))
    states = []
    for step_idx in range(total_steps):
        offset = step_idx * float(grid_step)
        pitches = {}
        valid = True
        for voice in VOICE_ORDER:
            midi_value = _active_pitch_at_offset(satb_parts[voice], offset)
            if midi_value is None:
                valid = False
                break
            low, high = VOICE_RANGES[voice]
            midi_value = int(max(low, min(high, midi_value)))
            pitches[voice] = midi_value
        if not valid:
            continue
        if not (pitches["bass"] < pitches["tenor"] < pitches["alto"] < pitches["soprano"]):
            continue
        states.append(tuple(pitches[voice] for voice in VOICE_ORDER))
    return states


def load_chorale_state_sequences(folder, cache_path, refresh_cache=False, grid_step=DEFAULT_GRID_STEP):
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    all_files = sorted(
        p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    cache = _load_cache(cache_path)
    cached_files = cache["files"]
    active_keys = {str(p.resolve()) for p in all_files}
    for key in list(cached_files.keys()):
        if key not in active_keys:
            del cached_files[key]

    sequences = []
    reused = 0
    for path in all_files:
        key = str(path.resolve())
        mtime_ns = path.stat().st_mtime_ns
        cached_entry = cached_files.get(key)
        if (
            not refresh_cache
            and cached_entry is not None
            and cached_entry.get("schema") == "direct_chorale_states_v1"
            and cached_entry.get("mtime_ns") == mtime_ns
            and float(cached_entry.get("grid_step", -1.0)) == float(grid_step)
        ):
            states = [tuple(int(v) for v in state) for state in cached_entry.get("states", [])]
            if states:
                sequences.append(states)
            reused += 1
            continue

        try:
            score = converter.parse(str(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        states = _extract_state_sequence_from_score(score, grid_step=grid_step)
        cached_files[key] = {
            "schema": "direct_chorale_states_v1",
            "mtime_ns": mtime_ns,
            "grid_step": float(grid_step),
            "states": [list(state) for state in states],
        }
        if states:
            sequences.append(states)

    _save_cache(cache_path, cache)
    if all_files:
        total_states = sum(len(seq) for seq in sequences)
        print(
            f"Direct chorale scan complete: {len(all_files)} files, {reused} loaded from cache, "
            f"{len(sequences)} usable sequences, {total_states} states total."
        )
    return sequences


class DirectChoraleMarkovModel:
    def __init__(
        self,
        states,
        initial_pair_probabilities,
        transition_matrix,
        laplace_alpha=1.0,
        grid_step=DEFAULT_GRID_STEP,
    ):
        self.states = list(states)
        self.initial_pair_probabilities = {
            (int(i), int(j)): float(prob)
            for (i, j), prob in initial_pair_probabilities.items()
        }
        self.transition_matrix = {
            (int(i), int(j)): {int(k): float(prob) for k, prob in probs.items()}
            for (i, j), probs in transition_matrix.items()
        }
        self.laplace_alpha = float(laplace_alpha)
        self.grid_step = float(grid_step)
        self._state_indexes = {state: idx for idx, state in enumerate(self.states)}

    @classmethod
    def train_from_sequences(cls, sequences, laplace_alpha=1.0, grid_step=DEFAULT_GRID_STEP):
        flat_states = [state for sequence in sequences for state in sequence]
        if len(flat_states) < 3:
            raise ValueError("Need at least three SATB states to train the direct chorale model.")

        states = sorted(set(flat_states))
        state_index = {state: idx for idx, state in enumerate(states)}
        initial_pair_counts = {}
        transition_counts = {}
        state_totals = {}

        for sequence in sequences:
            if len(sequence) < 2:
                continue
            first_idx = state_index[sequence[0]]
            second_idx = state_index[sequence[1]]
            initial_pair_counts[(first_idx, second_idx)] = (
                initial_pair_counts.get((first_idx, second_idx), 0.0) + 1.0
            )
            for left, middle, right in zip(sequence, sequence[1:], sequence[2:]):
                i = state_index[left]
                j = state_index[middle]
                k = state_index[right]
                pair = (i, j)
                if pair not in transition_counts:
                    transition_counts[pair] = {}
                transition_counts[pair][k] = transition_counts[pair].get(k, 0.0) + 1.0
                state_totals[pair] = state_totals.get(pair, 0.0) + 1.0

        initial_total = sum(initial_pair_counts.values())
        initial_pair_probabilities = {
            pair: count / initial_total for pair, count in initial_pair_counts.items()
        }

        num_states = len(states)
        uniform = 1.0 / num_states
        normalized_transition = {}
        for pair, next_counts in transition_counts.items():
            total = state_totals[pair] + (float(laplace_alpha) * num_states)
            probs = {state_idx: float(laplace_alpha) / total for state_idx in range(num_states)}
            for state_idx, count in next_counts.items():
                probs[state_idx] = (count + float(laplace_alpha)) / total
            normalized_transition[pair] = probs

        return cls(
            states=states,
            initial_pair_probabilities=initial_pair_probabilities,
            transition_matrix=normalized_transition,
            laplace_alpha=laplace_alpha,
            grid_step=grid_step,
        )

    def generate_states(self, length, seed_state=None):
        length = max(1, int(length))
        if length == 1:
            if seed_state is not None and tuple(seed_state) in self._state_indexes:
                return [tuple(seed_state)]
            state_scores = np.zeros(len(self.states), dtype=float)
            for (i, j), prob in self.initial_pair_probabilities.items():
                state_scores[i] += prob
                state_scores[j] += prob
            state_scores = state_scores / state_scores.sum()
            start_idx = int(np.random.choice(len(self.states), p=state_scores))
            return [self.states[start_idx]]

        if seed_state is not None and tuple(seed_state) in self._state_indexes:
            first = tuple(seed_state)
            first_idx = self._state_indexes[first]
            next_probs = np.zeros(len(self.states), dtype=float)
            for (i, j), prob in self.initial_pair_probabilities.items():
                if i == first_idx:
                    next_probs[j] += prob
            total = next_probs.sum()
            next_probs = (
                np.full(len(self.states), 1.0 / len(self.states))
                if total <= 0
                else next_probs / total
            )
            second_idx = int(np.random.choice(len(self.states), p=next_probs))
            second = self.states[second_idx]
        else:
            pairs = list(self.initial_pair_probabilities.keys())
            probs = np.array([self.initial_pair_probabilities[pair] for pair in pairs], dtype=float)
            probs = probs / probs.sum()
            pair_idx = int(np.random.choice(len(pairs), p=probs))
            first_idx, second_idx = pairs[pair_idx]
            first = self.states[first_idx]
            second = self.states[second_idx]

        out = [first, second]
        while len(out) < length:
            prev_idx = self._state_indexes[out[-2]]
            curr_idx = self._state_indexes[out[-1]]
            pair = (prev_idx, curr_idx)
            if pair in self.transition_matrix:
                probs_dict = self.transition_matrix[pair]
                probs = np.array(
                    [probs_dict.get(state_idx, 0.0) for state_idx in range(len(self.states))],
                    dtype=float,
                )
            else:
                probs = np.full(len(self.states), 1.0 / len(self.states))
            next_idx = int(np.random.choice(len(self.states), p=probs))
            out.append(self.states[next_idx])
        return out

    def generate_voices(self, length, seed_state=None):
        state_sequence = self.generate_states(length, seed_state=seed_state)
        voices = {voice: [] for voice in ["soprano", "alto", "tenor", "bass"]}
        reverse_order = ["soprano", "alto", "tenor", "bass"]
        index_map = {"bass": 0, "tenor": 1, "alto": 2, "soprano": 3}
        for voice in reverse_order:
            seq = [(int(state_sequence[0][index_map[voice]]), float(self.grid_step))]
            for state in state_sequence[1:]:
                pitch_value = int(state[index_map[voice]])
                if pitch_value == seq[-1][0]:
                    seq[-1] = (seq[-1][0], float(seq[-1][1] + self.grid_step))
                else:
                    seq.append((pitch_value, float(self.grid_step)))
            voices[voice] = seq
        return voices

    def save(self, model_path):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=np.array(self.states, dtype=object),
            initial_pair_probabilities=np.array([self.initial_pair_probabilities], dtype=object),
            transition_matrix=np.array([self.transition_matrix], dtype=object),
            laplace_alpha=np.array([self.laplace_alpha]),
            grid_step=np.array([self.grid_step]),
        )

    @classmethod
    def load(cls, model_path):
        with np.load(model_path, allow_pickle=True) as data:
            states = [tuple(int(v) for v in state) for state in data["states"].tolist()]
            return cls(
                states=states,
                initial_pair_probabilities=data["initial_pair_probabilities"][0],
                transition_matrix=data["transition_matrix"][0],
                laplace_alpha=float(data["laplace_alpha"][0]),
                grid_step=float(data["grid_step"][0]),
            )


def build_chorale_score(voices):
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Direct Chorale Markov Model")
    for voice_name in ["soprano", "alto", "tenor", "bass"]:
        part = stream.Part()
        part.id = voice_name.capitalize()
        for midi_value, duration in voices[voice_name]:
            n = note.Note(quarterLength=float(duration))
            n.pitch.midi = int(midi_value)
            part.append(n)
        score.append(part)
    return score


def visualize_chorale(voices):
    score = build_chorale_score(voices)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = f"direct_chorale_{timestamp}.musicxml"
    xml_path = score.write("musicxml", fp=output_file)
    try:
        os.startfile(xml_path)
        print(f"Saved and opened: {xml_path}")
    except OSError:
        print(f"Saved: {xml_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Direct chorale Markov baseline trained on SATB chorale states.")
    parser.add_argument("--data-folder", default=DEFAULT_DATA_FOLDER, help="Folder containing chorale files.")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Cache path for extracted SATB states.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Saved model path.")
    parser.add_argument("--grid-step", type=float, default=DEFAULT_GRID_STEP, help="Quantization step in quarter lengths.")
    parser.add_argument("--laplace-alpha", type=float, default=1.0, help="Laplace smoothing alpha.")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH, help="Number of grid states to generate.")
    parser.add_argument("--retrain", action="store_true", help="Retrain model from corpus.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild cached SATB state sequences.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)

    if args.retrain or not model_path.exists():
        reason = "--retrain was set" if args.retrain else "model file not found"
        print(f"Training direct chorale model because {reason}.")
        sequences = load_chorale_state_sequences(
            args.data_folder,
            cache_path=args.cache_path,
            refresh_cache=args.refresh_cache,
            grid_step=args.grid_step,
        )
        if not sequences:
            raise ValueError("No usable SATB chorale sequences were found for training.")
        model = DirectChoraleMarkovModel.train_from_sequences(
            sequences,
            laplace_alpha=args.laplace_alpha,
            grid_step=args.grid_step,
        )
        model.save(model_path)
        print(f"Saved trained model to '{model_path}'.")
    else:
        model = DirectChoraleMarkovModel.load(model_path)
        print(f"Loaded existing direct chorale model from '{model_path}'.")

    voices = model.generate_voices(args.length)
    visualize_chorale(voices)


if __name__ == "__main__":
    main()
