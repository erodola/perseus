"""Perseus on its Rust runtime (perseus-core) as an Encrypted GPT-2 Speedrun server: the harness protocol, driving
perseus.serve.gpt2_decode's handler (the served planned decode) instead of its socket.

    bash speedrun/serve.sh --track decode --inbox IN --outbox OUT [--device N]

The wire is the speedrun's (harness/wire.py, harness/protocol.py): every file is an 8-byte little-endian length and
the payload, written under a temporary name and renamed. Setup, untimed: compile the plan's programs from the CPU
twin, write the key request out/keyspec (pc_client::KeySpec JSON, docs/WIRE.md), wait for the client's bundle
(in/keys/keys.pckb, complete once in/keys.done exists), open the session on it (no secret), and write out/ready
with the level block 0 takes its input at. Decode-8: request k is the 16 positions in/k/0..15 from an empty cache,
each a ciphertext frame of the token row; position j answers out/k/j with [logits frame, CutMax z frame] in
speedrun/record1.py's list framing. Prefill-128 is not served yet. The session ends at in/quit.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import types

from speedrun.record1 import pack

HEADER = struct.Struct("<Q")
STEPS = 16


def write(root, rel, payload):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(len(payload)))
        f.write(payload)
    os.rename(tmp, path)


def read(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        data = f.read()
    return data[HEADER.size:HEADER.size + HEADER.unpack_from(data)[0]]


def wait(root, *rels):
    while True:
        for rel in rels:
            if os.path.exists(os.path.join(root, rel)):
                return rel
        time.sleep(0.0005)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=("prefill", "decode"), required=True)
    p.add_argument("--inbox", required=True)
    p.add_argument("--outbox", required=True)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    if a.track != "decode":
        raise SystemExit("server.py: Prefill-128 is not served on perseus-core yet")
    from perseus.examples.defaults import default_plan
    from perseus.client import contract_hash
    from perseus.serve import gpt2_decode as served
    args = types.SimpleNamespace(plan=default_plan("gpt2_decode"), chain="n32", layers=None, packing="cachemir_complex",
                                 weights=os.environ["WEIGHTS_PATH"], key_layout="packed", pt_rows=28, stc_cache=8,
                                 work_dir="/tmp/speedrun")
    h = served.Gpt2Handler(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)      # the handler exports its own device; the GPU is ours
    man = h.manifest({"tokens": STEPS, "argmax": True, "contract": contract_hash(served._config(args.plan, "n32"))})
    write(a.outbox, "keyspec", man["spec"].encode())
    wait(a.inbox, "keys.done")
    h.load(os.path.join(a.inbox, "keys", "keys.pckb"), None)
    write(a.outbox, "ready", json.dumps({"input": man["input"]}).encode())
    k = 0
    while wait(a.inbox, f"{k}/0", "quit") != "quit":
        for j in range(STEPS):
            wait(a.inbox, f"{k}/{j}")
            _h, outs = h.run({"t": j, "argmax": True}, [read(a.inbox, f"{k}/{j}")], None)
            write(a.outbox, f"{k}/{j}", pack(outs))
        k += 1
    h.run_.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
