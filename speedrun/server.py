"""Record 0's server for the Encrypted GPT-2 Speedrun: the harness protocol, from the client's key bundle alone.

    bash speedrun/serve.sh --track {prefill,decode} --inbox IN --outbox OUT [--device N] [--profile]

The wire is the speedrun's (harness/wire.py, harness/protocol.py): every file is an 8-byte little-endian length and
the payload, written under a temporary name and renamed. The payloads are speedrun/record0.py's. Setup, untimed: write
the key request out/keyspec (it names the track; record0.options fixes the keys), wait for the client's key bundle
(in/keys/, complete once in/keys.done exists), open the session from it (it holds no secret key), load the weights,
the plan and the bundle's PCMM keys, and write out/ready with the level the client encrypts fresh inputs at. The
prefill also runs one warm-up request on a public prompt (seeded token ids). Each request k then runs its steps
in/k/j -> out/k/j: Decode-8 the 16 positions from an empty cache, each answered with its logit tiles and CutMax
argmax; Prefill-128 the 128-token prompt through the blocks, answered with the last position's logit tiles and
argmax. The session ends at in/quit.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

from speedrun import record0

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
    p.add_argument("--track", choices=tuple(record0.PLANS), required=True)
    p.add_argument("--inbox", required=True)
    p.add_argument("--outbox", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--profile", action="store_true", help="per-step times on stderr (device syncs added)")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)
    write(a.outbox, "keyspec", json.dumps({"record0": a.track}).encode())   # the key set is record0.options'
    wait(a.inbox, "keys.done")
    keys, plan = os.path.join(a.inbox, "keys"), record0.PLANS[a.track]
    opts, runtime = record0.options(a.track, a.device)
    from examples.gpt2_from_primitives import env
    sess = env.open_session(a.device, options=opts, bundle=keys)
    from perseus import _core
    from perseus.impl import config
    from perseus.nn.serve import serialize_tiles
    from examples.gpt2_from_primitives import weights
    from examples.gpt2_from_primitives.model import Gpt2Model
    from examples.gpt2_from_primitives.prefill import Gpt2Prefill
    inf = sess.inf
    env.bootstrap_setup(inf.fhe, runtime, bundle=keys)
    store = weights.RawStore(os.environ["WEIGHTS_PATH"])
    if a.track == "decode":
        model = Gpt2Model(inf, store, config.load_configs(env.plan_configs(plan, "n32")), core=_core,
                          packing="cachemir_complex")
        model.load_plans(plan)
    else:
        model = Gpt2Prefill(inf, store, config.load_configs(env.plan_configs(plan, "n32", env.CONFIGS_N32_LEGACY)),
                            core=_core)
        model.load_plans(plan, argmax_blocks=False)
        model.preload_prefill()
        record0.load_keys(model.rt, a.track, keys)
    missing = env.rotation_audit(inf.fhe, model.rt.dims)
    if missing:
        raise SystemExit(f"server.py: rotation keys missing from the bundle: {missing}")

    def answer(tiles, pos):
        z = model.argmax_encrypted(tiles, pos)
        if a.profile:
            model.rt.ops.device_sync()
            t = time.time()
        out = record0.pack([serialize_tiles(inf, tiles, _core), serialize_tiles(inf, z, _core)])
        if a.profile:
            print(f"[speedrun] pos{pos} serialize={time.time() - t:.3f}s ({len(out) / 2**20:.1f} MiB)",
                  file=sys.stderr, flush=True)
        return out

    def prefill(Xs, ch):
        Xs, ch, _ = model.prefill_cts(Xs, ch)
        return answer(model.tail(Xs, ch), ch.T - 1)

    if a.track == "prefill":     # warm-up on a public prompt: masks, StC caches and pools; nothing of the client's
        ids = np.random.default_rng(0).integers(0, store.shape(weights.WTE)[0], record0.PROMPT)
        prefill(*model.encode(store.tensor(weights.WTE)[ids] + store.tensor(weights.WPE)[:record0.PROMPT]))
        model.rt.ops.device_sync()
        model.overlap = False    # as run_prefill's timed pass after its warm-up
    write(a.outbox, "ready", json.dumps({"bootstrap_output_level": int(inf.fhe.bootstrap_output_level())}).encode())
    k = 0
    while wait(a.inbox, f"{k}/0", "quit") != "quit":
        if a.track == "decode":
            model.start()
            for j in range(record0.STEPS["decode"]):
                wait(a.inbox, f"{k}/{j}")
                t0 = time.time()
                x = _core.adopt_input(inf, read(a.inbox, f"{k}/{j}"), True)
                t1 = time.time()
                tiles, _ = model.decode_step(x, j)
                if a.profile:
                    model.rt.ops.device_sync()
                t2 = time.time()
                write(a.outbox, f"{k}/{j}", answer(tiles, j))
                if a.profile:
                    print(f"[speedrun] pos{j} in={t1 - t0:.3f}s forward={t2 - t1:.3f}s "
                          f"argmax+out={time.time() - t2:.3f}s", file=sys.stderr, flush=True)
        else:
            Xs = [_core.adopt_input(inf, b, False) for b in record0.unpack(read(a.inbox, f"{k}/0"))]
            write(a.outbox, f"{k}/0", prefill(Xs, model.chunks(record0.PROMPT)))
        k += 1
    model.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
