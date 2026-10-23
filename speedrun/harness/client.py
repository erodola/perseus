"""The harness client as a process: one session of one track, its result written as JSON.

    python3 -I harness/client.py --track {prefill,decode} --inbox IN --outbox OUT --result FILE
                                 [--backend {dummy,NAME}] [--prompts FILE --ref FILE]
                                 [--requests N] [--offset K] [--warmup W] [--timeout S]

The harness owns the inputs and the checks. Request i takes prompt K + i of `--prompts` (prompts/dev.json, or the
hidden test set; K = `--offset`), its first 128 tokens for Prefill-128 and 16 for Decode-8; the float64 reference
(harness/gpt2_ref.py on `--ref`, HuggingFace's GPT-2 weights) gives the input rows and the reference logits. The
backend holds the crypto only: it writes the key bundle the server's key request asks for, encrypts the rows
before the request starts, and decrypts each output to (logits, argmax) after it ends, so none of it is timed.
The harness then takes the KL of every scored position and checks that the argmax is the logits' own; a request
passes with its mean KL under the track's bar and every scored argmax right. Backend NAME is the class `Backend` of
harness/NAME_client.py (`-` read as `_`): `perseus` is record 0's, `perseus-core` record 1's, each run under
harness/with_perseus.sh on its Perseus checkout. `dummy` sends random bytes and expects the dummy server's answer,
their SHA-256. The first `--warmup` requests run but are marked unscored. Bytes from the server never reach the
JSON, only digests and verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import TRACKS, Session  # noqa: E402

TOKENS = {"prefill": 128, "decode": 16}


class Dummy:
    def __init__(self, track: str):
        self.steps = TRACKS[track].steps

    def keys(self, path: str, keyspec: bytes) -> None:
        os.makedirs(path)

    def accept(self, manifest: bytes) -> None:
        pass

    def inputs(self, i: int) -> list[bytes]:
        return [os.urandom(1024) for _ in range(self.steps)]

    def check(self, inputs: list[bytes], outputs: list[bytes]) -> dict:
        return {"outputs_ok": all(o == hashlib.sha256(x).digest() for x, o in zip(inputs, outputs))}


class Checked:
    """A real backend's requests: the prompts in, the reference out, the checks on the decrypted outputs."""

    def __init__(self, backend, track: str, prompts: str, ref: str, offset: int = 0):
        from gpt2_ref import GPT2
        self.backend, self.track, self.offset = backend, track, offset
        with open(prompts, encoding="utf-8") as f:
            self.prompts = json.load(f)["prompts"]
        self.ref = GPT2(ref)

    def inputs(self, i: int) -> list[bytes]:
        self.prompt = (self.offset + i) % len(self.prompts)
        rows = self.ref.embed(self.prompts[self.prompt][:TOKENS[self.track]])
        logits = self.ref.logits(rows)
        self.want = [logits[-1]] if self.track == "prefill" else list(logits)
        return self.backend.encrypt(rows)

    def check(self, inputs: list[bytes], outputs: list[bytes]) -> dict:
        import numpy as np
        from gpt2_ref import kl
        got = self.backend.decrypt(outputs)
        kls = [kl(w, lg) for w, (lg, _) in zip(self.want, got)]
        argmax_ok = [int(am) == int(np.argmax(lg)) for lg, am in got]
        t = TRACKS[self.track]
        mean = float(np.mean([kls[j] for j in t.scored]))
        return {"prompt": self.prompt, "kl": kls, "kl_scored_mean": mean, "argmax_ok": argmax_ok,
                "outputs_ok": mean < t.kl_max and all(argmax_ok[j] for j in t.scored)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--inbox", required=True)
    p.add_argument("--outbox", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--backend", default="dummy")
    p.add_argument("--prompts")
    p.add_argument("--ref")
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--timeout", type=float, default=600.0)
    a = p.parse_args()
    session = Session(a.inbox, a.outbox, a.track, timeout=a.timeout)
    out = {"track": a.track, "backend": a.backend, "requests": []}
    try:
        if a.backend == "dummy":
            backend = runs = Dummy(a.track)
        else:
            backend = importlib.import_module(f"{a.backend.replace('-', '_')}_client").Backend(a.track)
            runs = Checked(backend, a.track, a.prompts, a.ref, a.offset)
            with open(a.prompts, "rb") as f:
                out["prompts_sha256"] = hashlib.sha256(f.read()).hexdigest()
        session.send_keys(backend.keys)
        manifest = session.ready()
        out["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        backend.accept(manifest)
        for i in range(a.warmup + a.requests):
            inputs = runs.inputs(i)
            r = session.request(inputs)
            out["requests"].append({"warmup": i < a.warmup, "times_s": r.times(), "score": r.score(a.track),
                                    **runs.check(inputs, r.outputs)})
    except Exception as e:  # noqa: BLE001 (the run's verdict, not a crash)
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        session.quit()
        with open(a.result, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 1 if "error" in out else 0


if __name__ == "__main__":
    sys.exit(main())
