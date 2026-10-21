# Magenta RealTime integration

The performance application can now send its generated notes to Magenta
RealTime 2 for audio accompaniment. Everything runs on one computer, but the
camera/music application and Magenta remain separate processes so model
inference cannot block hand tracking, MIDI timing, or the score display.

## Data path

```text
MediaPipe hands -> PerformanceSession -> GeneratedTick
                                      -> normal MIDI playback
                                      -> live score
                                      -> MagentaRouter (25 Hz)
                                           -> 127.0.0.1 UDP gateway
                                                -> mock logger, or
                                                -> Magenta RT Input MIDI port
                                                     -> MRT2 audio
```

The packets contain the 128-pitch conditioning state expected by MRT2:
`0=off`, `1=sustain`, and `2=onset`. The router follows independent voice
onsets, releases, held soprano notes, articulation gates, pause/play, and mode
changes. Magenta communication never runs on the musical clock thread.

## Test on the current computer

Open one terminal and start the mock gateway:

```bash
python Code/magenta_host.py --backend mock
```

Open another terminal and run the hand application:

```bash
python Code/hand_performance.py --dry-run --magenta
```

The gateway prints the pitches it receives. The performance window shows
`Connected (mock)` when the loopback path is working. `--dry-run` only disables
the existing synthesizer output; it does not disable Magenta packets.

## Run with Magenta RealTime 2

On a compatible Apple Silicon computer:

1. Install/build the official MRT2 standalone app and download its model files.
2. Start MRT2, load a model, and confirm that it created the virtual MIDI port
   named `Magenta RT Input`.
3. Start the local MIDI gateway:

   ```bash
   python Code/magenta_host.py --backend midi
   ```

4. Start this project's interface:

   ```bash
   python Code/hand_performance.py --magenta
   ```

Use **Magenta On/Off** in the application to enable accompaniment. Choose
**Soprano only** to condition Magenta with the performed/generated melody, or
**All four voices** to send the entire SATB texture. The official MRT2 app owns
the generated audio output and its prompt/style controls.

If the virtual port has a different name, pass part of it explicitly:

```bash
python Code/magenta_host.py --backend midi --midi-port "Magenta RT Input"
```

List available MIDI destinations with:

```bash
python Code/magenta_host.py --list-midi-ports
```

The gateway only binds to `127.0.0.1`; it does not accept traffic from another
computer. If the gateway or MRT2 stops, the existing Markov performance keeps
running and the UI changes to `Waiting for local gateway`.
