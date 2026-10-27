# Direct Magenta RealTime integration

The performance application sends generated notes directly to the virtual MIDI
input created by the official Magenta RealTime 2 standalone application. There
is no custom protocol, UDP connection, gateway, or second project process.

## Data path

```text
MediaPipe hands -> PerformanceSession -> GeneratedTick
                                      -> normal MIDI playback
                                      -> live score
                                      -> direct Magenta MIDI router
                                           -> Magenta RT Input
                                                -> MRT2 audio
```

The router follows independent voice onsets, releases, held soprano notes,
articulation gates, pause/play, and mode changes. **Soprano only** sends the
final generated or hand-performed soprano. **All four voices** sends the full
SATB texture. The official Magenta application converts these MIDI events into
its internal conditioning frames.

## Test without Magenta

The in-process mock backend prints the MIDI events that would be sent to MRT2:

```bash
python Code/hand_performance.py --dry-run --magenta --magenta-backend mock
```

The performance window shows `Connected (mock)`. `--dry-run` disables the
project's ordinary synthesizer output but does not disable the Magenta router.

## Run with Magenta RealTime 2

On a compatible Apple Silicon computer:

1. Start the official MRT2 standalone application.
2. Load a model and confirm that the application created a virtual MIDI input
   named `Magenta RT Input`.
3. Start the hand interface:

   ```bash
   python Code/hand_performance.py --dry-run --magenta --magenta-backend midi
   ```

The UI shows `Connected (Magenta RT Input)` after the port opens. If MRT2 is not
running, it reports that the port was not found and retries in the background.
The camera, Markov generator, live score, and other MIDI output continue to run.

Use **Magenta On/Off** in the application to enable accompaniment. Choose
**Soprano only** to guide Magenta with the melody or **All four voices** to send
the complete texture. The official MRT2 application owns the generated audio
and its prompt/style controls.

If the virtual port has a different name, pass part of that name directly:

```bash
python Code/hand_performance.py --dry-run --magenta --magenta-backend midi \
    --magenta-midi-port "Magenta RT Input"
```

Remove `--dry-run` to hear the project's regular MIDI output together with the
Magenta audio. In that case, use `--midi-port` to select a separate synthesizer
so the ordinary player does not also target Magenta's virtual port.
