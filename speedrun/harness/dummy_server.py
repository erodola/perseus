"""A stand-in server for the protocol tests: no FHE, CPU only.

    python3 -I harness/dummy_server.py --track {prefill,decode} --inbox IN --outbox OUT
                                       [--delay S] [--fill-delay S]
                                       [--misbehave {early,silent,linger,poke}]

It answers each input with its SHA-256 after `--delay` seconds (`--fill-delay` for decode positions
0-7), and exits 3 if the client released an input before the previous output was written. `early`
writes an output before its input exists; `silent` never answers; `linger` ignores `quit`; `poke`
exits 4 if it can write its own inbox (it must not, in the sandbox).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402
from protocol import TRACKS  # noqa: E402

LIFETIME = 900.0  # seconds; a server orphaned by a failed test still exits


def next_request(inbox: str, k: int, deadline: float) -> bool:
    """Block until request k starts (True) or the client quits (False)."""
    while time.monotonic() < deadline:
        if wire.exists(inbox, "quit"):
            return False
        if wire.exists(inbox, f"{k}/0"):
            return True
        time.sleep(0.0005)
    raise wire.WireError("no request before the deadline")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--inbox", required=True)
    p.add_argument("--outbox", required=True)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--fill-delay", type=float, default=None)
    p.add_argument("--misbehave", choices=("early", "silent", "linger", "poke"))
    a = p.parse_args()
    deadline = time.monotonic() + LIFETIME
    if a.misbehave == "poke":
        try:
            wire.write(a.inbox, "poked", b"")
            print("dummy_server: the inbox is writable", file=sys.stderr)
            return 4
        except OSError:
            pass
    wire.write(a.outbox, "keyspec", b"{}")      # a dummy needs no keys
    wire.wait(a.inbox, "keys.done", deadline)
    wire.write(a.outbox, "ready", f'{{"server": "dummy", "track": "{a.track}"}}'.encode())
    if a.misbehave == "early":
        wire.write(a.outbox, "0/0", b"guess")
    k = 0
    while next_request(a.inbox, k, deadline):
        for j in range(TRACKS[a.track].steps):
            rel = f"{k}/{j}"
            wire.wait(a.inbox, rel, deadline)
            payload = wire.read(a.inbox, rel, cap=1 << 28)
            fill = a.track == "decode" and j < 8 and a.fill_delay is not None
            time.sleep(a.fill_delay if fill else a.delay)
            if a.misbehave == "silent":
                continue
            if wire.exists(a.inbox, f"{k}/{j + 1}"):
                print(f"dummy_server: {k}/{j + 1} released before output {rel}", file=sys.stderr)
                return 3
            wire.write(a.outbox, rel, hashlib.sha256(payload).digest())
        k += 1
    if a.misbehave == "linger":
        time.sleep(max(0.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
