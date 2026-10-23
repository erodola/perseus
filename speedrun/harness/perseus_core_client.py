"""Record 1's client crypto (Perseus on its Rust runtime, perseus-core): pc-client through perseus._pc, no GPU.

Used by harness/client.py --backend perseus-core, which harness/with_perseus.sh runs with that checkout's toolchain.
The server's key request is a pc_client::KeySpec (that checkout's crates/pc-client/src/wire.rs): the client generates exactly those
keys, after checking that the request is for the chain it knows (params_hash). Decode-8 sends each token row as a
ciphertext frame (pack_token_rows at the level the server announces); Prefill-128 sends the prompt's blocked slot
vectors (perseus.impl.blocked) as frames at the (level, degree) the server lists. Every output holds the logits and
the CutMax z frames of one position (speedrun/record1.py's framing), which `decrypt` turns into (logits, argmax).
The harness owns the rest. The client refuses a key request for keys record 1 does not declare (`refuse`): the server
chooses the request, and a sparse secret is safe only at the modulus its declaration names.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

PERSEUS = "89bc830017fe12ab967ddb10c24feae65d8af6df"     # record 1: the Perseus commit this client runs (runner/run.py)
VOCAB = 50257
SPRU_H = 64                                            # the SPRU block key: a single 1 in each of 64 blocks
SPARSE = ("enc_dense_to_sparse", "spru_atob")          # the switching keys to a sparse secret (Hamming weight 32, SPRU)
BOTTOM = 1                                             # n32: their limbs, q0 and q1 (the bottom level) and P


def parse(keyspec: bytes) -> dict:
    """The server's key request as pc-client reads it: a repeated JSON key, which two parsers may resolve differently,
    is refused."""
    def unique(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError(f"a repeated key in the key request: {[k for k, _ in pairs]}")
        return dict(pairs)
    return json.loads(keyspec, object_pairs_hook=unique)


def refuse(spec: dict) -> str | None:
    """Why the client will not generate the keys `spec` asks for (None if it will): only record 1's declared keys,
    each at 128 bits where it is used (README Rule 2). Its sparse secrets live at the bottom modulus, the SPRU block
    key has 64 blocks (2 would leave 2^15 candidates), and CCMM keys (ring 2^13) are not declared."""
    if spec.get("chain") != "n32":
        return f"chain {spec.get('chain')!r}, not n32"
    spru = spec.get("spru")
    if spru is not None and (spru.get("h") != SPRU_H or spru.get("n") != 2):
        return f"a SPRU block key of {spru.get('h')} blocks, not {SPRU_H}"
    if spec.get("ccmm"):
        return "CCMM keys, which record 1 does not declare"
    for k in spec.get("keys", []):
        if k.get("role") in SPARSE and not k.get("shape", {}).get("q_top", BOTTOM + 1) <= BOTTOM:
            top = k.get("shape", {}).get("q_top")
            return f"a {k['role']} key up to limb {top}, above the bottom level (limb {BOTTOM})"
    return None


class PerseusCore:
    def __init__(self, track: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the client never touches a GPU
        from perseus.client import Client, ClientSecret, pack_token_rows
        from perseus.impl.layout import Dims
        from speedrun import record1
        self.r1, self.pack_rows, self.d, self.track = record1, pack_token_rows, Dims.gpt2(), track
        self.client = Client(ClientSecret.generate(), complex=True)
        self.level = self.metas = None

    def keys(self, path: str, keyspec: bytes) -> None:
        spec = parse(keyspec)
        if spec.get("params_hash") != self.client.params_hash:
            raise RuntimeError("the server's key request is for another chain than the client's")
        if why := refuse(spec):
            raise RuntimeError(f"the server's key request asks for {why}")
        os.makedirs(path)
        self.client.keygen(keyspec.decode(), os.path.join(path, "keys.pckb"), threads=8)

    def accept(self, manifest: bytes) -> None:
        m = json.loads(manifest)
        if self.track == "decode":
            self.level = int(m["input"]["level"])
        else:
            self.metas = [(int(lv), int(dg)) for lv, dg in m["inputs"]]

    def encrypt(self, rows) -> list[bytes]:
        import numpy as np
        d = self.d
        if self.track == "decode":
            return [bytes(self.client.encrypt(self.pack_rows([r], d.N, d.hid, d.dim), self.level)) for r in rows]
        from perseus.impl.blocked import Blocks, blocked_slots
        from perseus.impl.prefill import Chunks
        B = Blocks(d, self.r1.NCH)
        vals = [v for blk in blocked_slots(np.asarray(rows), Chunks(len(rows), B.t), B) for v in blk]
        if len(vals) != len(self.metas):
            raise RuntimeError(f"{len(vals)} blocked inputs, the server takes {len(self.metas)}")
        return [self.r1.pack([bytes(self.client.encrypt(v, lv, dg)) for v, (lv, dg) in zip(vals, self.metas)])]

    def decrypt(self, outputs: list[bytes]) -> list[tuple]:
        import numpy as np
        from perseus.examples.gpt2.head import decode_logits, decode_z
        shim = SimpleNamespace(dims=self.d,
                               ops=SimpleNamespace(decrypt_slots_complex=lambda f: self.client.decrypt(f)[0]))
        out = []
        for payload in outputs:
            lg, z = self.r1.unpack(payload, most=2)
            z = decode_z(shim, [z], VOCAB, self.d.N)
            out.append((decode_logits(shim, [lg], VOCAB, self.d.N), int(np.argmax(z))))
        return out


Backend = PerseusCore
