"""Record 0's client crypto on the pinned Perseus checkout: keygen, encryption and decryption, with no GPU.

Used by harness/client.py --backend perseus, which harness/with_perseus.sh runs with that checkout's toolchain.
The keys, layouts and payloads are record 0's (that checkout's speedrun/record0.py). Decode-8 sends each token row
encrypted; Prefill-128 sends the 128-token prompt as its blocked ciphertexts. Every output holds the logit tiles and
the CutMax argmax of one position, which `decrypt` turns into (logits, argmax). The harness owns the rest.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

PERSEUS = "39493d5a98122c1fb595c99070bf4e435df752f0"     # record 0 (tag speedrun-record-0): the commit it runs
VOCAB = 50257


class Perseus:
    def __init__(self, track: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the client never touches a GPU
        from speedrun import record0
        self.r0, self.track = record0, track
        opts, _ = record0.options(track, "")
        from perseus import _core
        from perseus.impl.layout import Dims
        from perseus.nn.remote import EncClient
        self.core = _core
        self.client = EncClient(options=opts, family="gpt2")
        inf = self.client.inf
        self.rt = SimpleNamespace(dims=Dims.gpt2(), ops=SimpleNamespace(
            decrypt_slots_complex=lambda t: np.asarray(_core.decrypt_slots_complex(inf, t), dtype=np.complex128),
            decode_linear_output=lambda t, d_in, d_out: np.asarray(
                _core.decode_linear_output(inf, t, int(d_in), int(d_out)), dtype=np.float64)))

    def keys(self, path: str, keyspec: bytes) -> None:
        """Record 0's key set is fixed by record0.options: the server's request only names the track."""
        self.client.save_bundle(path)
        self.r0.save_keys(self.client, self.track, path)

    def accept(self, manifest: bytes) -> None:
        self.client.accept(json.loads(manifest))

    def encrypt(self, rows) -> list[bytes]:
        if self.track == "decode":
            return [self.client.encrypt(r) for r in rows]
        from perseus.impl.blocked import blocked_slots
        from perseus.impl.prefill import Chunks
        B, level = self.r0.blocks(), int(self.client.encode_level)
        slots = blocked_slots(rows, Chunks(len(rows), B.t), B)
        return [self.r0.pack([self.core.encrypt_slots(self.client.inf, v, level) for blk in slots for v in blk])]

    def decrypt(self, outputs: list[bytes]) -> list[tuple]:
        from perseus.nn.serve import deserialize_tiles
        from examples.gpt2_from_primitives.head import decode_logits, decode_z
        W, out = self.rt.dims.N, []
        for payload in outputs:
            tiles, z = (deserialize_tiles(self.client.inf, b, self.core) for b in self.r0.unpack(payload, most=2))
            out.append((decode_logits(self.rt, tiles, VOCAB, W), int(np.argmax(decode_z(self.rt, z, VOCAB, W)))))
        return out


Backend = Perseus
