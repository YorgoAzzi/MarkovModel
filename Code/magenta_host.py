"""Local gateway from the hand application to Magenta RealTime 2.

Use ``--backend mock`` on any computer to inspect the conditioning stream. On
an Apple Silicon computer, start the official MRT2 standalone app and use
``--backend midi`` to forward the stream to its ``Magenta RT Input`` port.
"""

from __future__ import annotations

import argparse
import socket
import time

try:
    from magenta_protocol import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        MIDI_ONSET,
        MagentaStatePacket,
        make_ack,
    )
except ModuleNotFoundError:
    from .magenta_protocol import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        MIDI_ONSET,
        MagentaStatePacket,
        make_ack,
    )


class MockBackend:
    name = "mock"

    def __init__(self, log_every: float = 1.0):
        self.log_every = max(0.0, float(log_every))
        self.last_log = 0.0
        self.frames = 0
        self.last_packet: MagentaStatePacket | None = None

    def handle(self, packet: MagentaStatePacket) -> None:
        self.frames += 1
        self.last_packet = packet
        now = time.monotonic()
        if self.log_every and now - self.last_log >= self.log_every:
            sounding = [pitch for pitch, state in enumerate(packet.notes) if state]
            print(
                f"frame={packet.sequence:06d} enabled={packet.enabled} "
                f"input={packet.input_mode} notes={sounding} tempo={packet.tempo_bpm:.1f}"
            )
            self.last_log = now

    def close(self) -> None:
        return


class MidiBackend:
    """Forward conditioning notes to the official MRT2 virtual MIDI port."""

    name = "midi"

    def __init__(self, port_name: str = "Magenta RT Input", mido_module=None):
        if mido_module is None:
            try:
                import mido as mido_module
            except ImportError as exc:
                raise RuntimeError("Install mido and python-rtmidi for MIDI forwarding") from exc
        self.mido = mido_module
        names = self.mido.get_output_names()
        matches = [name for name in names if port_name.lower() in name.lower()]
        if not matches:
            raise RuntimeError(
                f"Magenta MIDI port containing {port_name!r} was not found. "
                f"Start the MRT2 standalone app first. Available ports: {names}"
            )
        self.port_name = matches[0]
        self.port = self.mido.open_output(self.port_name)
        self.active: set[int] = set()
        print(f"Magenta MIDI output: {self.port_name}")

    def _send(self, kind: str, pitch: int, velocity: int = 0) -> None:
        self.port.send(
            self.mido.Message(kind, channel=0, note=int(pitch), velocity=int(velocity))
        )

    def stop_all(self) -> None:
        for pitch in sorted(self.active):
            self._send("note_off", pitch, 0)
        self.active.clear()

    def handle(self, packet: MagentaStatePacket) -> None:
        if not packet.enabled:
            self.stop_all()
            return
        desired = {pitch for pitch, state in enumerate(packet.notes) if state}
        onsets = {pitch for pitch, state in enumerate(packet.notes) if state == MIDI_ONSET}
        for pitch in sorted(self.active - desired):
            self._send("note_off", pitch, 0)
            self.active.remove(pitch)
        for pitch in sorted(desired):
            if pitch in onsets and pitch in self.active:
                self._send("note_off", pitch, 0)
                self.active.remove(pitch)
            if pitch not in self.active:
                self._send("note_on", pitch, packet.velocity)
                self.active.add(pitch)

    def close(self) -> None:
        self.stop_all()
        self.port.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Local gateway from the Markov hand app to Magenta RealTime"
    )
    parser.add_argument("--backend", choices=("mock", "midi"), default="mock")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--midi-port", default="Magenta RT Input")
    parser.add_argument("--log-every", type=float, default=1.0)
    parser.add_argument("--list-midi-ports", action="store_true")
    return parser.parse_args(argv)


def run_server(backend, *, port: int = DEFAULT_PORT, stop_event=None) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((DEFAULT_HOST, int(port)))
    if stop_event is not None:
        server.settimeout(0.2)
    print(f"Magenta local gateway: {DEFAULT_HOST}:{port} ({backend.name})")
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                payload, address = server.recvfrom(8192)
            except socket.timeout:
                continue
            except ConnectionResetError:
                # Windows reports an ICMP port-unreachable response if a
                # client closes before reading the gateway acknowledgement.
                continue
            try:
                packet = MagentaStatePacket.from_bytes(payload)
                backend.handle(packet)
                status = "ready"
            except Exception as exc:
                print(f"Ignored invalid conditioning packet: {exc}")
                continue
            server.sendto(make_ack(packet.sequence, backend.name, status), address)
    except KeyboardInterrupt:
        print("Stopping Magenta gateway")
    finally:
        backend.close()
        server.close()


def main(argv=None):
    args = parse_args(argv)
    if args.list_midi_ports:
        import mido

        for name in mido.get_output_names():
            print(name)
        return
    backend = (
        MockBackend(args.log_every)
        if args.backend == "mock"
        else MidiBackend(args.midi_port)
    )
    run_server(backend, port=args.port)


if __name__ == "__main__":
    main()
