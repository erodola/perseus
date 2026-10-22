"""Record 0 of the Encrypted GPT-2 Speedrun: what its server (speedrun/server.py) and the harness client's crypto
(harness/perseus_client.py, run on this checkout) agree on. The plans, the session options both sides build from (the
client generates every key under them, the server loads them), the prefill's PCMM keys, and the payloads: a list of
byte strings is a u32 count, the u64 lengths, then the strings (little-endian). A Decode-8 input is one encrypted token
row; a Prefill-128 input is the list of the prompt's blocked ciphertexts; an output, in both tracks, is the list
[logit tiles, CutMax argmax tiles] (each perseus.nn.serve.serialize_tiles).
"""
from __future__ import annotations

import os
import struct

import numpy as np

from examples.gpt2_from_primitives import env

PLANS = {"decode": "bootstrap_placements/gpt2_decode_python_n32", "prefill": "bootstrap_placements/gpt2_prefill128"}
STEPS = {"decode": 16, "prefill": 1}
PROMPT = 128
NCH = 512                                    # the prefill's channel blocks (PF_NCH)
_COUNT, _LEN = struct.Struct("<I"), struct.Struct("<Q")


def pack(blobs: list[bytes]) -> bytes:
    return _COUNT.pack(len(blobs)) + b"".join(_LEN.pack(len(b)) for b in blobs) + b"".join(blobs)


def unpack(payload: bytes, most: int = 64) -> list[bytes]:
    """pack's inverse; refuses a malformed payload (the client reads it from the server)."""
    if len(payload) < _COUNT.size:
        raise ValueError("record0.unpack: no payload header")
    (n,) = _COUNT.unpack_from(payload)
    off = _COUNT.size + n * _LEN.size
    if n > most or off > len(payload):
        raise ValueError("record0.unpack: bad payload header")
    lens = [_LEN.unpack_from(payload, _COUNT.size + i * _LEN.size)[0] for i in range(n)]
    if off + sum(lens) != len(payload):
        raise ValueError("record0.unpack: lengths do not match the payload")
    out = []
    for m in lens:
        out.append(payload[off:off + m])
        off += m
    return out


def runtime(track):
    """Export the track's plan switches as run_decode / run_prefill do; returns the session overrides. Call it before
    importing perseus.impl: modules read some switches at import."""
    plan = PLANS[track]
    if track == "decode":
        return {**env.plan_runtime(plan, "n32"), "GPT2_PACKING": "cachemir_complex"}
    os.environ.setdefault("KV_ARENA_GB", "0")
    fast = {"FIDESLIB_ENCODE_NOSYNC": "1", "FIDESLIB_CCMM_ASYNC": "1", "FIDESLIB_PCMM_NOSYNC": "1",
            "FIDESLIB_ENCODE_PINNED_RING": "256", "FIDESLIB_STC_POW2_KEY": "1"}
    over = {**{k: os.environ.get(k, v) for k, v in fast.items()}, **env.plan_runtime(plan, "n32"),
            "GPT2_PACKING": "cachemir"}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    cal = os.path.join(plan, "pcmm_calib.json")
    if os.path.exists(cal):
        os.environ.setdefault("PF_PCMM_CALIB_IN", cal)
    return {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}


def blocks():
    from perseus.impl.blocked import Blocks
    from perseus.impl.layout import Dims
    return Blocks(Dims.gpt2(), NCH)


def options(track, device):
    """(the session options, the overrides): what the client generates the keys under and the server opens with.
    The prefill adds its plan's rotation steps and those of the tail (the last row into the decode layout, the LM
    head) to the decode's."""
    over = runtime(track)
    extra = set()
    if track == "prefill":
        from perseus.impl.blocked import token_of_rot_steps
        from perseus.impl.layout import linear_rot_steps
        from perseus.impl.magtape import plan_rot_steps
        B = blocks()
        extra = set(plan_rot_steps(PLANS[track]) or ()) | token_of_rot_steps(B, (PROMPT - 1) % B.t) \
            | linear_rot_steps(B.d.N, B.d.hid, B.d.N)
    opts = env.session_options(device, complex_payload=True, chain="n32", extra_rot_steps=extra, **over)
    return opts, over


def save_keys(client, track, path):
    """The key owner's half of the prefill's PCMM keys (perseus.impl.pcmm.make_keys, from the secret key and the
    chain's 4 lowest moduli) as pcmm_keys.npz. The decode needs nothing past the bundle."""
    if track != "prefill":
        return
    from perseus import _core
    from perseus.impl.pcmm import make_keys
    fhe = client.inf.fhe
    keys = make_keys(_core.bts.secret_key_coeffs(fhe), _core.bts.chain_moduli(fhe, 4), blocks().t)
    np.savez(os.path.join(path, "pcmm_keys.npz"), **{k: np.asarray(v) for k, v in keys.items()})


def load_keys(rt, track, path):
    """The server's half: the PCMM engine of the prefill's lanes gets the bundle's keys (it never makes its own)."""
    if track != "prefill":
        return
    from perseus import _core
    from perseus.impl.pcmm import engine
    with np.load(os.path.join(path, "pcmm_keys.npz")) as z:
        engine(rt, blocks().t).keys = _core.bts.pcmm_keys(dict(z))
