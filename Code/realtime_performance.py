"""Run the movement-controlled engine against a MIDI output in real time."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

from realtime_engine import RealtimeChoraleEngine


VOICE_CHANNELS = {"bass": 0, "tenor": 1, "alto": 2, "soprano": 3}


class MidiPlayer:
    def __init__(self, port_name=None):
        try:
            import mido
        except ImportError as exc:
            raise RuntimeError("Install mido and python-rtmidi for live MIDI playback") from exc
        self.mido = mido
        names = mido.get_output_names()
        if not names:
            raise RuntimeError("No MIDI output ports found. Use --dry-run or connect a MIDI synthesizer.")
        if port_name:
            matches = [name for name in names if port_name.lower() in name.lower()]
            if not matches:
                raise RuntimeError(f"MIDI port containing {port_name!r} was not found: {names}")
            selected = matches[0]
        else:
            selected = names[0]
        self.port = mido.open_output(selected)
        self.active = {}
        for channel in VOICE_CHANNELS.values():
            self.port.send(mido.Message("program_change", channel=channel, program=52))
        print(f"MIDI output: {selected}")

    def render(self, frame) -> None:
        expression = max(1, min(127, int(10 + frame.controls.volume * 117)))
        for channel in VOICE_CHANNELS.values():
            self.port.send(self.mido.Message("control_change", channel=channel, control=11, value=expression))
        if not frame.new_notes:
            return
        for voice, channel in VOICE_CHANNELS.items():
            old_pitch = self.active.get(channel)
            if old_pitch is not None:
                self.port.send(self.mido.Message("note_off", channel=channel, note=old_pitch, velocity=0))
            pitch = int(frame.pitches[voice])
            self.port.send(
                self.mido.Message("note_on", channel=channel, note=pitch, velocity=frame.velocity)
            )
            self.active[channel] = pitch

    def close(self) -> None:
        for channel, pitch in self.active.items():
            self.port.send(self.mido.Message("note_off", channel=channel, note=pitch, velocity=0))
        self.port.close()


class NullPlayer:
    def render(self, frame) -> None:
        return

    def close(self) -> None:
        return


def _read_commands(command_queue, stop_event):
    while not stop_event.is_set():
        try:
            command_queue.put(input("movement> ").strip())
        except (EOFError, KeyboardInterrupt):
            command_queue.put("quit")
            return


def _apply_command(engine, text, stop_event):
    parts = text.split()
    if not parts:
        return
    command = parts[0].lower()
    if command in {"quit", "exit", "stop", "q"}:
        stop_event.set()
    elif command == "accent":
        engine.update_controls(accent=True)
        print("Queued accent for the next note attack.")
    elif command in {"density", "volume", "tilt", "rotation"} and len(parts) == 2:
        engine.update_controls(**{command: float(parts[1])})
        target = engine.controls.target()
        print(f"Set {command} target to {getattr(target, command):.2f}.")
    elif command == "controls" and len(parts) == 5:
        engine.update_controls(
            density=float(parts[1]),
            volume=float(parts[2]),
            tilt=float(parts[3]),
            rotation=float(parts[4]),
        )
        print(f"Set control targets to {engine.controls.target()}.")
    elif command == "key" and len(parts) == 2:
        engine.configure(key=parts[1])
        print(f"Key set to {engine.key}; harmony updates at the next bar.")
    elif command == "mode" and len(parts) == 2:
        engine.configure(mode=parts[1])
        print(f"Mode set to {engine.mode}; harmony updates at the next bar.")
    elif command == "status":
        print(f"target controls: {engine.controls.target()}")
    elif command == "help":
        print_help()
    else:
        print("Unknown command. Type 'help' for available controls.")


def print_help():
    print("Commands can be entered while music continues:")
    print("  density 0..1 | volume 0..1 | tilt -1..1 | rotation 0..1 | accent")
    print("  controls <density> <volume> <tilt> <rotation>")
    print("  key <C/G/F#...> | mode <mode> | status | help | quit")


def parse_args():
    parser = argparse.ArgumentParser(description="Low-latency movement-controlled SATB performance")
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--ticks", type=int, default=0, help="Stop after N ticks; 0 runs until quit")
    parser.add_argument("--midi-port", default=None, help="Substring of the MIDI output port name")
    parser.add_argument("--dry-run", action="store_true", help="Run the clock and generator without MIDI")
    parser.add_argument("--list-midi-ports", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=4, help="Print timing every N ticks; 0 disables")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_midi_ports:
        import mido

        for name in mido.get_output_names():
            print(name)
        return

    engine = RealtimeChoraleEngine.from_paths(tempo_bpm=args.tempo, seed=args.seed)
    player = NullPlayer() if args.dry_run else MidiPlayer(args.midi_port)
    commands = queue.Queue()
    stop_event = threading.Event()
    print_help()
    if sys.stdin.isatty():
        reader = threading.Thread(target=_read_commands, args=(commands, stop_event), daemon=True)
        reader.start()
    elif args.ticks <= 0:
        raise RuntimeError("Non-interactive runs require --ticks so the performance has an endpoint")

    deadline = time.perf_counter()
    generated = 0
    try:
        while not stop_event.is_set() and (args.ticks <= 0 or generated < args.ticks):
            while True:
                try:
                    _apply_command(engine, commands.get_nowait(), stop_event)
                except queue.Empty:
                    break
                except (TypeError, ValueError) as exc:
                    print(f"Invalid control: {exc}")

            started = time.perf_counter()
            frame = engine.generate_tick()
            generation_ms = (time.perf_counter() - started) * 1000.0
            player.render(frame)
            if args.log_every > 0 and generated % args.log_every == 0:
                lateness_ms = max(0.0, (time.perf_counter() - deadline) * 1000.0)
                chord = f"{frame.chord[0]}:{frame.chord[1]}"
                print(
                    f"tick={frame.tick_index:04d} bar={frame.bar_index + 1:02d} "
                    f"new_notes={int(frame.new_notes)} chord={chord} generation={generation_ms:.1f}ms "
                    f"scheduler_late={lateness_ms:.1f}ms "
                    f"controls={frame.controls.density:.2f}/{frame.controls.volume:.2f}/"
                    f"{frame.controls.tilt:.2f}/{frame.controls.rotation:.2f} "
                    f"rhythm={frame.sampled_duration_beats:g}beat"
                )
            generated += 1
            deadline += engine.tick_seconds
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                stop_event.wait(remaining)
            else:
                # Recover cleanly instead of accumulating missed deadlines.
                deadline = time.perf_counter()
    finally:
        player.close()
        print("Realtime performance stopped.")


if __name__ == "__main__":
    main()
