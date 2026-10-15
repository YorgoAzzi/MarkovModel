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

TensorFlow is not used by this project. If an older installation causes a
TensorFlow DLL error while MediaPipe starts, remove it and reinstall the project
dependencies:

```bash
python -m pip uninstall tensorflow
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

The original CLI-driven realtime mode generates and exports one bar at a time:

```bash
python "Code/markovchain.py" --task realtime --realtime-bars 16
```

Realtime mode writes `generated_realtime_live.musicxml` after each bar.

### Low-latency movement prototype

`Code/realtime_performance.py` runs a persistent SATB engine on a 16th-note
clock. Movement controls are smoothed and can affect the next clock tick while
the learned chord progression and voice-leading state continue across ticks.

List MIDI outputs:

```bash
python "Code/realtime_performance.py" --list-midi-ports
```

Play through the first available MIDI output:

```bash
python "Code/realtime_performance.py" --tempo 120
```

Test generation and timing without a synthesizer:

```bash
python "Code/realtime_performance.py" --dry-run --ticks 64 --seed 7
```

Commands can be entered without stopping the playback clock:

- `density 0.8` shortens generated note durations so the rhythm moves faster.
- `volume 0.6` changes MIDI loudness without changing the rhythm.
- `tilt -0.5` moves the voices toward a lower register.
- `rotation 0.7` encourages wider motion and delays cadences.
- `accent` accents the next generated attack.
- `controls 0.8 0.6 -0.5 0.7` sets density, volume, tilt, and rotation together.
- `key G`, `mode dorian`, `status`, and `quit` are also available.

The engine itself is in `Code/realtime_engine.py`. Its `update_controls()` and
`generate_tick()` methods are the integration points for a future phone sensor
receiver; the phone connection does not need to know about the Markov internals.

Realtime rhythm uses the trained second-order duration Markov model. Density
biases its next-duration probabilities: low density favors longer learned
durations and high density favors shorter ones. The result remains sampled, so
the same movement can produce rhythmic variation instead of a fixed duration.
If density rises during a long note, a dynamic duration limit lets the engine
respond without waiting for the entire previously sampled note.

### Colored-ribbon camera control

The camera prototype tracks a brightly colored ribbon or band. Movement speed
controls density, and vertical position controls volume:

- Move slowly for longer, less dense rhythms; move quickly for shorter, denser rhythms.
- Move down for quieter music; move up for louder music.

Start it with:

```bash
python "Code/camera_performance.py" --tempo 120
```

Click the ribbon once in the camera window to calibrate its color, then move it.
The camera controls only density and volume; horizontal position, acceleration,
tilt, rotation, key, and mode do not affect the music.

Useful calibration options include `--hue-tolerance`, `--min-area`, and
`--max-speed`. Lower `--max-speed` if density reacts too weakly; raise it if
density reaches maximum too easily. Press `Q` or Escape to stop. Use `--dry-run`
to test tracking without sending MIDI.

### MediaPipe two-hand control

The two-hand interface uses MediaPipe landmarks while OpenCV continues to
capture and display the webcam image. The official hand model is stored at
`models/hand_landmarker.task`. It now has two modes that can be changed without
restarting the performance.

**Full SATB mode** keeps the original interface:

- Right-hand height controls the generated register.
- Left-hand height controls melodic activity.
- Left-hand horizontal speed controls tempo; maximum tempo is reached at a
  deliberately slow default movement speed of 0.20 screen widths per second.
- Left-hand openness controls articulation from staccato to legato.
- Right-hand finger patterns select scale degrees 1--7.
- Right lean selects major, upright uses the diatonic quality, and left lean
  selects minor.

The chord gestures are: index=1, index+middle=2,
index+middle+ring=3, index+middle+ring+pinky=4, all five fingers=5,
thumb=6, and thumb+index=7. A fist or any other pattern holds the last accepted
chord.

**Soprano Performance mode** changes the musical roles:

- The Markov/voice-leading engine generates alto, tenor, and bass only.
- Moving the right hand upward or downward steps successively through the
  selected key and mode for the soprano, independently of the current chord.
- A stationary right hand holds one continuous soprano note; moving vertically
  replaces it immediately with the next scale note.
- A quick horizontal right-hand shake starts a trill.
- A quick index-finger bend and release plays a mordent.
- The number of extended fingers changes only the soprano timbre: one=piano,
  two=violin, three=choir, four=flute, and five=synth pad.
- A thumb-index pinch sustains the soprano, and a fist rests it.
- Left-hand height controls the register of alto, tenor, and bass; horizontal
  speed controls tempo; openness controls articulation.
- Chord gestures are disabled in this mode.

Forward/backward brightness control has been removed from the hand interface.

Start the interface with:

```bash
python "Code/hand_performance.py" --tempo 60 --max-tempo 140 --key C --mode major
```

Click **Full SATB** or **Soprano Performance** at the bottom of the camera
window to switch modes. Click **Pause/Play** to stop or resume generation while
hand tracking continues. Keyboard shortcuts are `1`, `2`, and Space. You can
start directly in the second mode with `--performance-mode soprano`.

The hand interface runs the musical clock at 16 ticks per beat. Soprano scale
steps can therefore reach the scheduler on the next fine tick. A stationary
note has no scheduled MIDI note-off; the next scale step, ornament, fist, pause,
or mode change ends it immediately. Use `--soprano-step` to change how much
vertical travel produces one scale step, `--pinch-distance` to tune pinch
detection, `--trill-shake-speed` to tune the trill, and `--timbre-dwell` to
control how long a finger count must remain stable before the sound changes.

### Live sheet music

The PyQt performance window places the camera and live SATB score side by side.
Soprano and alto appear on the treble staff, tenor and bass on the bass staff,
and the current sonority is red. The active section contains four measures by
default and updates whenever the generator starts new notes. Completed sections
remain in the scrollable score panel. The display follows the newest section
until you scroll upward; the **Follow live** button returns to the current music.

Music playback does not wait for notation. Generated onsets enter a queue, and
a background worker converts them through `music21` to MusicXML and Verovio to
SVG. The Qt interface renders those SVG pages directly inside the application.

Use `--score-bars 6` to show six measures in each score section or
`--no-live-score` to hide the score panel.

Use `--dry-run` to test tracking without MIDI. Hand labels are swapped by
default for the current mirrored-camera setup; add `--no-swap-hands` if the
physical hands are already labeled correctly. Chord gestures must remain
stable briefly and are applied on beat boundaries.
Extended fingertips are shown in green and folded fingertips in red. If a
normally extended finger remains red, lower `--finger-angle` slightly (for
example, to `125`); raise it if folded fingers are being shown as extended.
The thumb also uses its distance from the palm. If a tucked thumb remains green,
raise `--thumb-spread`; lower it if an open thumb remains red. Lower
`--max-horizontal-speed` if slow lateral movement does not raise tempo enough.
Height reaches its full range between 0.25 and 0.75 of the camera image; tune
those limits with `--height-top` and `--height-bottom`.

## Notes

- First retrain can be slow due to symbolic parsing.
- Caching is enabled automatically; retrains become faster.
- Use `--refresh-cache` when source datasets change significantly.
