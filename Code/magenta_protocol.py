"""Small loopback protocol used by the performance app and Magenta gateway."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable


PROTOCOL_NAME = "markov-magenta"
PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765
DEFAULT_FRAME_RATE = 25.0

INPUT_SOPRANO = "soprano"
INPUT_ALL = "all"
INPUT_MODES = (INPUT_SOPRANO, INPUT_ALL)

MIDI_OFF = 0
MIDI_SUSTAIN = 1
MIDI_ONSET = 2
MIDI_MODEL_DECIDES = 3
MIDI_STATE_VALUES = {MIDI_OFF, MIDI_SUSTAIN, MIDI_ONSET, MIDI_MODEL_DECIDES}


def _validated_notes(values: Iterable[int]) -> tuple[int, ...]:
    notes = tuple(int(value) for value in values)
    if len(notes) != 128:
        raise ValueError("A Magenta MIDI state must contain exactly 128 pitches")
    if any(value not in MIDI_STATE_VALUES for value in notes):
        raise ValueError("MIDI states must be 0 (off), 1 (sustain), 2 (onset), or 3")
    return notes


@dataclass(frozen=True)
class MagentaStatePacket:
    """One frame of the 128-pitch MIDI conditioning signal."""

    sequence: int
    sent_at_ns: int
    enabled: bool
    input_mode: str
    notes: tuple[int, ...]
    velocity: int = 90
    tempo_bpm: float = 120.0

    def __post_init__(self):
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown Magenta input mode: {self.input_mode}")
        object.__setattr__(self, "notes", _validated_notes(self.notes))
        object.__setattr__(self, "velocity", max(1, min(127, int(self.velocity))))
        object.__setattr__(self, "tempo_bpm", max(1.0, float(self.tempo_bpm)))

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "protocol": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "type": "state",
                "sequence": int(self.sequence),
                "sent_at_ns": int(self.sent_at_ns),
                "enabled": bool(self.enabled),
                "input_mode": self.input_mode,
                "notes": list(self.notes),
                "velocity": self.velocity,
                "tempo_bpm": self.tempo_bpm,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "MagentaStatePacket":
        data = json.loads(payload.decode("utf-8"))
        if data.get("protocol") != PROTOCOL_NAME:
            raise ValueError("Packet is not part of the Markov-Magenta protocol")
        if int(data.get("version", -1)) != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported protocol version: {data.get('version')}")
        if data.get("type") != "state":
            raise ValueError(f"Expected a state packet, got {data.get('type')!r}")
        return cls(
            sequence=int(data["sequence"]),
            sent_at_ns=int(data["sent_at_ns"]),
            enabled=bool(data["enabled"]),
            input_mode=str(data["input_mode"]),
            notes=tuple(data["notes"]),
            velocity=int(data.get("velocity", 90)),
            tempo_bpm=float(data.get("tempo_bpm", 120.0)),
        )


def make_ack(sequence: int, backend: str, status: str = "ready") -> bytes:
    return json.dumps(
        {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "type": "ack",
            "sequence": int(sequence),
            "backend": str(backend),
            "status": str(status),
        },
        separators=(",", ":"),
    ).encode("utf-8")


def parse_ack(payload: bytes) -> dict:
    data = json.loads(payload.decode("utf-8"))
    if data.get("protocol") != PROTOCOL_NAME or data.get("type") != "ack":
        raise ValueError("Invalid Magenta acknowledgement")
    if int(data.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError("Unsupported acknowledgement version")
    return data
