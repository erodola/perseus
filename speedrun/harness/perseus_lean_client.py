"""The lean-chain entry's client crypto (Perseus on its Rust runtime): pc-client through perseus._pc, no GPU.

Used by harness/client.py --backend perseus-lean, which harness/with_perseus.sh runs with that checkout's toolchain.
It is harness/perseus_core_client.py for two chains, whichever the server's key request names: n32 (record 1's), or
n32a4 (pc-core `ChainSpec::n32a4`: n32's primes without three of its bottom StC primes, dnum 4, and P = n32's 9
special primes followed by 4 more). Every key under the dense ternary secret lives at Q x P (log QP 1706 on n32, 1734
on n32a4; at most 1747 at N = 2^16). The two keys to sparse secrets (the h = 32 encapsulation key and the SPRU block
key) are switching keys at the bottom over n32's 9 special primes on either chain, so their modulus q0 x P is record
1's exactly. The client generates the keys the request lists, after checking that it is for the chain it names
(params_hash) and that it holds only declared keys at their declared shapes (`refuse`).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

PERSEUS = "24d4b3fd04c67293524aac354cc290d02ae2647b"     # the Perseus commit this client runs (runner/run.py)
CHAINS = ("n32", "n32a4")
VOCAB = 50257
SPRU_H = 64                                            # the SPRU block key: a single 1 in each of 64 blocks
SPARSE = ("enc_dense_to_sparse", "spru_atob")          # the switching keys to a sparse secret (Hamming weight 32, SPRU)
BOTTOM = 1                                             # their Q limbs: q0 and q1 (the bottom level)
P_SPARSE = 9                                           # and their special primes: n32's 9 (record 1's modulus q0 x P)


def parse(keyspec: bytes) -> dict:
    """The server's key request as pc-client reads it: a repeated JSON key, which two parsers may resolve differently,
    is refused."""
    def unique(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError(f"a repeated key in the key request: {[k for k, _ in pairs]}")
        return dict(pairs)
    return json.loads(keyspec, object_pairs_hook=unique)


def refuse(spec: dict) -> str | None:
    """Why the client will not generate the keys `spec` asks for (None if it will): only the declared keys, each at
    128 bits where it is used (README Rule 2). The sparse secrets live at q0 x P' with P' n32's 9 special primes, the
    SPRU block key has 64 blocks, and CCMM keys (ring 2^13) are not declared. (pc-client also refuses a sparse key
    whose shape is not the chain's bottom shape.)"""
    if spec.get("chain") not in CHAINS:
        return f"chain {spec.get('chain')!r}, not one of {CHAINS}"
    spru = spec.get("spru")
    if spru is not None and (spru.get("h") != SPRU_H or spru.get("n") != 2):
        return f"a SPRU block key of {spru.get('h')} blocks, not {SPRU_H}"
    if spec.get("ccmm"):
        return "CCMM keys, which this entry does not declare"
    for k in spec.get("keys", []):
        if k.get("role") in SPARSE:
            shape = k.get("shape", {})
            if not shape.get("q_top", BOTTOM + 1) <= BOTTOM:
                return f"a {k['role']} key up to limb {shape.get('q_top')}, above the bottom level (limb {BOTTOM})"
            if not shape.get("kp", P_SPARSE + 1) <= P_SPARSE:
                return f"a {k['role']} key over {shape.get('kp')} special primes, more than n32's {P_SPARSE}"
    return None


class PerseusLean:
    def __init__(self, track: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the client never touches a GPU
        from perseus.impl.layout import Dims
        from speedrun import record1
        from perseus.client import pack_token_rows
        self.r1, self.pack_rows, self.d, self.track = record1, pack_token_rows, Dims.gpt2(), track
        self.client = self.level = self.metas = None

    def keys(self, path: str, keyspec: bytes) -> None:
        from perseus.client import Client, ClientSecret
        spec = parse(keyspec)
        if why := refuse(spec):
            raise RuntimeError(f"the server's key request asks for {why}")
        self.client = Client(ClientSecret.generate(spec["chain"]), chain=spec["chain"], complex=True)
        if spec.get("params_hash") != self.client.params_hash:
            raise RuntimeError(f"the server's key request is for another chain than {spec['chain']}")
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


Backend = PerseusLean
