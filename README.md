# Markov Symbolic Music Generator

This repository generates symbolic music from MIDI/MusicXML using Markov models. The two main workflows are:

- generate a new SATB chorale
- harmonize an input melody into SATB

Main script:
- `Code/markovchain.py`

Parameter reference:
- `Code/PARAMETERS.md`

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `music21` once so score files open correctly:

```bash
python -m music21.configure
```

## Data

Training folders:
- Melody/rhythm/HMM emission corpus: `data/` or `dataMelody/`
- Chord/progression corpus: `dataChords/`

Supported formats:
- `.mid`, `.midi`, `.xml`, `.musicxml`, `.mxl`

## Main Workflows

### 1. Generate A Chorale

This is the main generation mode. It creates a four-part SATB chorale using:
- chord progression generation
- chorale beam search
- learned rhythm templates
- cadence constraints

Default run:
```bash
python "Code/markovchain.py"
```

Equivalent explicit command:
```bash
python "Code/markovchain.py" --task song --song-style chorale --song-bars 16 --beats-per-bar 4 --key C --mode major
```

Higher-quality search:
```bash
python "Code/markovchain.py" --task song --song-style chorale --song-bars 16 --beats-per-bar 4 --chorale-beam-width 24 --chorale-candidates-per-voice 7 --chorale-top-sonorities 30
```

Example with different key and stronger cadence/repeat-note control:
```bash
python "Code/markovchain.py" --task song --song-style chorale --song-bars 16 --beats-per-bar 4 --key A --mode major --chorale-repeat-note-penalty 6 --cadence-every-bars 4
```

Notes:
- progression blocks are learned from `dataChords`
- chorale rhythm is learned from the melody corpus
- output is written as `generated_song_*.musicxml`

### 2. Harmonize An Input Melody

This mode takes an existing melody and generates SATB harmony around it.

Run with a direct file path:
```bash
python "Code/markovchain.py" --task harmonize --melody-input "path/to/melody.musicxml" --key C --mode major
```

Or place the file in `melodyInput/` and pass only the filename:
```bash
python "Code/markovchain.py" --task harmonize --melody-input "my_melody.musicxml"
```

If you omit `--melody-input`, the script picks the first supported file found in `melodyInput/`.

Important:
- if the melody is not actually in the selected key/mode, harmonization can fail
- for chromatic melodies, either use the correct `--key` / `--mode` or pass `--disable-scale-snap`
- multi-track MIDI input is resolved by choosing the highest-average-pitch note track as the melody line

Output:
- harmonized SATB score as `generated_song_*.musicxml`

## Common Options For Chorale And Harmonize

- `--key`, `--mode`: harmonic scale / key
- `--disable-scale-snap`: allow notes/chords outside the selected scale
- `--beats-per-bar`: event grid density
- `--cadence-every-bars`: cadence frequency
- `--chorale-beam-width`: beam width for SATB search
- `--chorale-candidates-per-voice`: per-voice candidate pool size
- `--chorale-top-sonorities`: local SATB expansion size
- `--chorale-repeat-note-penalty`: penalize repeated notes in a voice
- `--disable-progression-blocks`: fall back to raw chord-chain generation

## Other Modes

These are available, but secondary to the chorale and harmonize workflows.

### Melody

Train:
```bash
python "Code/markovchain.py" --task melody --retrain
```

Generate:
```bash
python "Code/markovchain.py" --task melody
```

### Chords

Train:
```bash
python "Code/markovchain.py" --task chord --retrain --refresh-cache
```

Generate:
```bash
python "Code/markovchain.py" --task chord --length 16
```

### Realtime

```bash
python "Code/markovchain.py" --task realtime --realtime-bars 16
```

Realtime mode writes `generated_realtime_live.musicxml` after each bar.

## Notes

- First retrain can be slow due to symbolic parsing.
- Caching is enabled automatically; retrains become faster.
- Use `--refresh-cache` when source datasets change significantly.
