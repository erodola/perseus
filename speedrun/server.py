"""Record 0's server for the Encrypted GPT-2 Speedrun: the harness protocol, from the client's key bundle alone.

    bash speedrun/serve.sh --track {prefill,decode} --inbox IN --outbox OUT [--device N]

The wire is the speedrun's (harness/wire.py, harness/protocol.py): every file is an 8-byte little-endian length and
the payload, written under a temporary name and renamed. Setup, untimed: wait for the client's key bundle (in/keys/,
complete once in/keys.done exists), open the session from it (it holds no secret key), load the weights and the plan,
and write out/ready with the level the client encrypts fresh inputs at. Decode-8: request k is the 16 positions
in/k/0..15 from an empty cache, and position j answers out/k/j with an 8-byte length, its logit tiles, then its CutMax
argmax tiles (perseus.nn.serve.serialize_tiles payloads). The session ends at in/quit.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

from examples.gpt2_from_primitives import env

PLAN = "bootstrap_placements/gpt2_decode_python_n32"
HEADER = struct.Struct("<Q")


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
    """Block until one of root/rels exists; return it."""
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
        raise SystemExit("server.py: Prefill-128 comes with speedrun step 4c")
    wait(a.inbox, "keys.done")
    keys = os.path.join(a.inbox, "keys")
    runtime = {**env.plan_runtime(PLAN, "n32"), "GPT2_PACKING": "cachemir_complex"}
    sess = env.open_session(a.device, complex_payload=True, chain="n32", bundle=keys, **runtime)
    from perseus import _core
    from perseus.impl import config
    from perseus.nn.serve import serialize_tiles
    from examples.gpt2_from_primitives import weights
    from examples.gpt2_from_primitives.model import Gpt2Model
    inf = sess.inf
    env.bootstrap_setup(inf.fhe, runtime, bundle=keys)
    model = Gpt2Model(inf, weights.RawStore(os.environ["WEIGHTS_PATH"]),
                      config.load_configs(env.plan_configs(PLAN, "n32")), core=_core, packing="cachemir_complex")
    missing = env.rotation_audit(inf.fhe, model.rt.dims)
    if missing:
        raise SystemExit(f"server.py: rotation keys missing from the bundle: {missing}")
    model.load_plans(PLAN)
    write(a.outbox, "ready", json.dumps({"bootstrap_output_level": int(inf.fhe.bootstrap_output_level())}).encode())
    k = 0
    while wait(a.inbox, f"{k}/0", "quit") != "quit":
        model.start()
        for j in range(16):
            wait(a.inbox, f"{k}/{j}")
            tiles, _ = model.decode_step(_core.deserialize_ct(inf, read(a.inbox, f"{k}/{j}")), j)
            z = model.argmax_encrypted(tiles, j)
            logits = serialize_tiles(inf, tiles, _core)
            write(a.outbox, f"{k}/{j}", HEADER.pack(len(logits)) + logits + serialize_tiles(inf, z, _core))
        k += 1
    model.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
