"""The protocol both tracks speak over the wire, and the clock the harness keeps.

A session starts with the server's key request, `keyspec` (which keys its plan needs, in its
backend's terms), and the client's key bundle for it: the client writes it under `in/keys/`, then
`in/keys.done`. The server's untimed setup (loading the keys, weights, plans) ends when it writes
`ready` with its session manifest. Only then does the client send requests, one at a time,
numbered from 0. A request is a sequence of steps: the client writes the input of step j at
`in/<k>/<j>`, the server answers at `out/<k>/<j>`, and input j+1 is released only after the client
has read output j. Prefill-128 has one step: the encrypted 128-token prompt in, the encrypted
last-position logits out. Decode-8 has 16, from an empty cache: one teacher-forced token in, its
encrypted logits out. The client ends the session with `in/quit`, and the server exits.

The clock is the client's. A step runs from the moment the client starts writing its input to the
moment its output appears. Prefill-128 scores its one step. Decode-8 scores positions 8-15 as one
span, from the release of token 8 to output 15, divided by 8, so work the server does between steps
is timed too. Positions 0-7 are reported, not scored. Payloads are opaque bytes at this layer.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import wire


@dataclass(frozen=True)
class Track:
    steps: int
    scored: range
    kl_max: float        # the bar on the KL from float64 GPT-2 over the scored positions


TRACKS = {"prefill": Track(steps=1, scored=range(0, 1), kl_max=1e-2),
          "decode": Track(steps=16, scored=range(8, 16), kl_max=1e-3)}


@dataclass
class Result:
    """One request: its outputs, and the monotonic start and end of every step."""
    outputs: list[bytes]
    starts: list[float]
    ends: list[float]

    def times(self) -> list[float]:
        return [e - s for s, e in zip(self.starts, self.ends)]

    def score(self, track: str) -> float:
        """Prefill-128: the request time (s). Decode-8: the time per scored token (s/token)."""
        scored = TRACKS[track].scored
        return (self.ends[scored[-1]] - self.starts[scored[0]]) / len(scored)


class Session:
    """The client side of one session with one server, over an inbox it writes and an outbox it reads."""

    def __init__(self, inbox: str, outbox: str, track: str, cap: int = 1 << 28, timeout: float = 600.0):
        self.inbox, self.outbox, self.track = inbox, outbox, TRACKS[track]
        self.cap, self.timeout = cap, timeout
        self.manifest: bytes | None = None
        self.k = 0

    def send_keys(self, write_bundle, timeout: float = 3600.0, cap: int = 1 << 24) -> None:
        """Wait for the server's key request; have `write_bundle(dir, keyspec)` write the bundle for it into
        `in/keys/`, then signal that it is complete."""
        wire.wait(self.outbox, "keyspec", time.monotonic() + timeout)
        write_bundle(os.path.join(self.inbox, "keys"), wire.read(self.outbox, "keyspec", cap))
        wire.write(self.inbox, "keys.done", b"")

    def ready(self, timeout: float = 3600.0, cap: int = 1 << 20) -> bytes:
        """Wait for the end of the server's setup; return its session manifest."""
        wire.wait(self.outbox, "ready", time.monotonic() + timeout)
        self.manifest = wire.read(self.outbox, "ready", cap)
        return self.manifest

    def request(self, inputs: list[bytes]) -> Result:
        """Run one request, releasing each input only after the previous output has been read."""
        if self.manifest is None:
            raise RuntimeError("request before the server is ready")
        if len(inputs) != self.track.steps:
            raise ValueError(f"{len(inputs)} inputs for a track of {self.track.steps} steps")
        k, self.k = self.k, self.k + 1
        result = Result([], [], [])
        for j, payload in enumerate(inputs):
            rel = f"{k}/{j}"
            if wire.exists(self.outbox, rel):
                raise wire.WireError(f"{rel}: output before its input")
            start = time.monotonic()
            wire.write(self.inbox, rel, payload)
            result.ends.append(wire.wait(self.outbox, rel, start + self.timeout))
            result.starts.append(start)
            result.outputs.append(wire.read(self.outbox, rel, self.cap))
        return result

    def quit(self) -> None:
        wire.write(self.inbox, "quit", b"")
