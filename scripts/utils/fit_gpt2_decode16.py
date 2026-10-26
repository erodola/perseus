#!/usr/bin/env python
"""Decode-8's approximation config (gpt2_base_n32_kc16): softmax and GELU fitted to positions 0-15 of 16-token windows.

Data: OpenWebText (Skylion007/openwebtext, parquet shard 0042) and C4's validation shard 00001 (allenai/c4), document
starts and random offsets, run through float64 GPT-2. Per layer, the score range over at most 16 keys widened by 15 %
of its width above and 5 % below sets the THOR rounds (delta2, delta1) and the exp series; the first reciprocal round's
band covers the data's head sums x3 and, at the top, every query whose scores lie in the range; the Chebyshev-scaled
Goldschmidt counts (SM_CHEB_GS) come from impl.smcheb.design over kc = 1..16. GELU's interval is 1.3 x the largest input.
The LayerNorm and CutMax sections are gpt2_base_n32's. The shipped file came from the same data through the decode's
plaintext mirror (another LayerNorm summation order): this script reproduces its choices and its numbers to ~1e-15.
The data files are pinned (OWT, C4 below: the dataset revision and the file's sha256; a file given by --owt / --c4 is
checked against the same sha256), and GPT-2 is openai-community/gpt2 (its HF cache snapshot unless --gpt2).

    python scripts/utils/fit_gpt2_decode16.py --out python/perseus/configs/model/approximation/gpt2_base_n32_kc16/configs.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "speedrun", "harness"))
from perseus.impl import smcheb                       # noqa: E402
from gpt2_ref import load_safetensors                 # noqa: E402

BASE = os.path.join(REPO, "python/perseus/configs/model/approximation/gpt2_base_n32/configs.json")
KMAX = 16
SOURCE = "16-token windows (document starts + random offsets) of OpenWebText shard 0042 and C4 validation shard 00001"
# (dataset, file, revision, sha256): the files the shipped config came from (OWT's parquet ref moves: its commit)
OWT = ("Skylion007/openwebtext", "plain_text/train/0042.parquet", "433fe0f44ed7894fea29c08b3202aa348ccc6369",
       "e1c54ab75be2c5ca6ed81313eade1615c478cbb2236cf0ddc86b29a3ac93fec1")
C4 = ("allenai/c4", "en/c4-validation.00001-of-00008.json.gz", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
      "db4c23d7a336032de37dfe2f3ed39653c8afd94004d4e8d9c73c03222299292c")
# the reciprocal rounds' counts the plans favour (same accuracy targets, another split): layer -> counts
SPLITS = {7: (6, 4, 4)}
# GELU degree per block, by hand: the lowest of 63 / 95 / 255 whose error at 1.3 x the data's max stays within ~2x the
# record's (95, not 127: the same BSGS depth with fewer products)
GELU_DEGREES = (63, 95, 255, 95, 63, 63, 63, 63, 63, 63, 95, 255)


def data_file(spec, path=None):
    """The pinned data file: `path` if given, else downloaded at the pinned revision; its sha256 checked either way."""
    repo, name, revision, sha256 = spec
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != sha256:
        raise SystemExit(f"{path}: sha256 {h.hexdigest()}, not {repo} {name}'s {sha256} (revision {revision})")
    return path


def windows(texts, seed, n_docs, chars, min_tokens, tok):
    """(document starts, random offsets): 16-token windows of the first n_docs shuffled documents with min_tokens."""
    rng = np.random.default_rng(seed)
    start, rand = [], []
    for i in rng.permutation(len(texts))[:n_docs]:
        ids = tok.encode(texts[i][:chars]).ids
        if len(ids) < min_tokens:
            continue
        start.append(ids[:KMAX])
        o = int(rng.integers(KMAX, len(ids) - KMAX))
        rand.append(ids[o:o + KMAX])
    return np.array(start, np.int64), np.array(rand, np.int64)


def ln(x, g, b, eps=1e-5):
    c = x - x.mean(-1, keepdims=True)
    return c / np.sqrt((c * c).mean(-1, keepdims=True) + eps) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def taps(w, ids, scores, gelu_max):
    """float64 GPT-2 on ids [B, 16]: every layer's score rows per cache length and each block's max |GELU input|."""
    x = w["wte.weight"][ids] + w["wpe.weight"][:ids.shape[1]]
    B, T, D = x.shape
    for i in range(12):
        p = f"h.{i}."
        h = ln(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"]) @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = (a.reshape(B, T, 12, 64).transpose(0, 2, 1, 3) for a in np.split(h, 3, axis=-1))
        s = q @ k.transpose(0, 1, 3, 2) / 8.0
        att = np.zeros_like(s)
        for t in range(T):
            row = s[:, :, t, :t + 1]
            scores[i].setdefault(t + 1, []).append(row.copy())
            e = np.exp(row - row.max(-1, keepdims=True))
            att[:, :, t, :t + 1] = e / e.sum(-1, keepdims=True)
        x = x + (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        u = ln(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"]) @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]
        gelu_max[i] = max(gelu_max[i], float(np.abs(u).max()))
        x = x + gelu(u) @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]


def next_pow2(x):
    return 1 if x <= 1 else 1 << math.ceil(math.log2(x))


def cheb_exp_series(lo, hi, degree, min_nodes=200):
    """calibrate.numerics.cheb_exp_series (without its torch import)."""
    n = max(8 * (degree + 1), min_nodes)
    k = np.arange(1, n + 1)
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k - 1) * np.pi / (2 * n))
    series = np.polynomial.chebyshev.Chebyshev.fit(x, np.exp(x), degree, domain=[lo, hi])
    return (series.coef / float(series(0.0))).astype(np.float64)


def split_targets(lo, hi, rounds, counts, tgt):
    """The first mid-round targets (smcheb.GRID order, widened) whose counts are `counts`."""
    for mids in itertools.product((0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7), repeat=rounds):
        tg = list(mids) + [tgt]
        ks = [smcheb.k_for(lo, hi, tg[0])] + [max(smcheb.k_for(*smcheb.refine_band(kc, tg[j - 1]), tg[j])
                                                  for kc in range(1, KMAX + 1)) for j in range(1, rounds + 1)]
        if tuple(ks) == tuple(counts):
            return tg
    raise ValueError(f"no target split gives counts {counts}")


def fit_softmax(rows, base, counts=None, hi_margin=0.15, lo_margin=0.05, cap=2885.0, d0=5.0, safety=3.0, tgt=1e-3):
    allv = np.concatenate([r.ravel() for r in rows.values()])
    a0, b0 = float(allv.min()), float(allv.max())
    a, b = a0 - lo_margin * (b0 - a0), b0 + hi_margin * (b0 - a0)
    M, mid = b - a, 0.5 * (a + b)
    delta2 = min(max(next_pow2(M / (2.0 * math.log(cap))), 2), 32)
    delta1 = min(max(next_pow2(M / (d0 * delta2)), 1), 64)
    delta0 = M / (delta1 * delta2)
    D0 = np.concatenate([np.exp((r - mid) / delta2).mean(-1).ravel() for r in rows.values()]
                        + [np.array([math.exp(-mid / delta2)])])          # the padded heads' scores are 0
    lo = float(D0.min()) / safety
    hi = max(float(D0.max()) * safety, 1.05 * math.exp((b - mid) / delta2))
    ia, ib = smcheb.cheb_seed(lo, hi)
    R = int(round(math.log2(delta2)))
    ks, tg = smcheb.design(ia, ib, R, tgt_last=tgt, kcs=tuple(range(1, KMAX + 1)))
    if counts is not None:
        ks, tg = list(counts), split_targets(*smcheb.band_from_seed(ia, ib), R, counts, tgt)
    coeffs = cheb_exp_series(-0.5 * delta0, 0.5 * delta0, 8)
    refine = smcheb.cheb_seed(*smcheb.refine_band(KMAX, 0.3))
    cfg = dict(base)
    cfg.pop("poly_coeffs", None)
    cfg.update({
        "mid_x": mid, "input_scale": float(delta1 * delta2), "squeeze_bound": 0.5 * M,
        "clip_lo": a, "clip_hi": b, "n_squarings": int(round(math.log2(delta1))), "refinement_iters": R,
        "cheb_coeffs": coeffs.tolist(), "cheb_a": -0.5 * delta0, "cheb_b": 0.5 * delta0,
        "init_alpha": ia, "init_beta": ib, "refine_alpha": [refine[0]] * R, "refine_beta": [refine[1]] * R,
        "gs_iters_scaled": int(ks[0]), "gs_iters_first": int(ks[0]), "gs_iters_refine_scaled": int(max(ks[1:], default=0)),
        "per_step_refine_iters": [int(k) for k in ks[1:]], "sm_kc_r": [1.0] * (R * KMAX),
        "recip_iters_cheb": [int(k) for k in ks], "recip_targets": [float(t) for t in tg],
        "delta0": delta0, "delta1": int(delta1), "delta2": int(delta2), "decode_window": KMAX, "calib_T": KMAX,
        "note": (f"Decode-8 geometry (<= {KMAX} keys): scores over {SOURCE} span [{a0:.2f}, {b0:.2f}], range "
                 f"-{lo_margin:g} / +{hi_margin:g} of that width; round-0 band from the data's head sums "
                 f"[{float(D0.min()):.3g}, {float(D0.max()):.3g}] x{safety:g}, its top at least 1.05 x the head sum of "
                 f"a query at the range top; SM_CHEB_GS counts (impl.smcheb.design, kc 1..{KMAX}, last target {tgt:g})"),
        "poly_coeffs": np.polynomial.chebyshev.cheb2poly(coeffs).tolist(),     # unused under cheb_coeffs
    })
    return cfg


def fit_gelu(xmax, degree, base, safety=1.3, eps=1e-9):
    """GELU(X t) / X on [-1, 1] at `degree`, X = safety * xmax (GELU_FOLD), reweighted toward minimax."""
    X = safety * xmax
    m = max(8 * (degree + 1), 4000)
    t = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    f = gelu(X * t) / X
    c = np.polynomial.chebyshev.chebfit(t, f, degree)
    for _ in range(6):
        r = np.abs(np.polynomial.chebyshev.chebval(t, c) - f)
        c = np.polynomial.chebyshev.chebfit(t, f, degree, w=np.sqrt(r / r.max() + 0.02))
    tt = np.linspace(-1, 1, 20001)
    e = float(np.abs(np.polynomial.chebyshev.chebval(tt, c) - gelu(X * tt) / X).max())
    g = dict(base)
    g.update({"xmax": X, "cheb_coeffs": c.tolist(), "cheb_a": -1.0, "cheb_b": 1.0, "cheb_eps": eps,
              "note": (f"degree-{degree} Chebyshev of GELU(X t)/X on [-1,1], X = {safety:g} x max|x| {xmax:.4g} over "
                       f"{SOURCE} (positions 0-15); max err {e:.1e} (x units {e * X:.1e})")})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--owt", help="Skylion007/openwebtext plain_text/train/0042.parquet (downloaded if absent; must "
                    "match OWT's sha256)")
    ap.add_argument("--c4", help="allenai/c4 en/c4-validation.00001-of-00008.json.gz (downloaded if absent; must match "
                    "C4's sha256)")
    ap.add_argument("--gpt2", help="GPT-2's model.safetensors and tokenizer.json directory (default: the HF cache's "
                    "openai-community/gpt2 snapshot)")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    hub = os.path.join(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "hub")
    snap = a.gpt2 or os.path.dirname(sorted(glob.glob(os.path.join(
        hub, "models--openai-community--gpt2", "snapshots", "*", "model.safetensors")))[0])
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    w = load_safetensors(os.path.join(snap, "model.safetensors"))
    owt = pq.read_table(data_file(OWT, a.owt), columns=["text"]).column("text").to_pylist()
    c4 = [json.loads(line)["text"] for line in gzip.open(data_file(C4, a.c4), "rt")]
    o_start, o_rand = windows(owt, 1234, 6000, 4000, 160, tok)
    c_start, c_rand = windows(c4, 8765, 12000, 3000, 128, tok)
    scores, gelu_max = [dict() for _ in range(12)], [0.0] * 12
    for ids in (np.concatenate([o_start[:1500], o_rand[:1500]]), np.concatenate([o_start[1500:], o_rand[1500:]]),
                np.concatenate([c_start[:5000], c_rand[:5000]])):
        for s0 in range(0, len(ids), 50):
            taps(w, ids[s0:s0 + 50], scores, gelu_max)
        print(f"[fit] {len(ids)} windows", flush=True)
    cfg = json.load(open(BASE))
    for i in range(12):
        name = f"transformer.h.{i}.attn"
        rows = {kc: np.concatenate(r, 0) for kc, r in scores[i].items()}
        cfg["softmax"][name] = fit_softmax(rows, cfg["softmax"][name], SPLITS.get(i))
        act = f"transformer.h.{i}.mlp.act"
        cfg["softgelu"][act] = fit_gelu(gelu_max[i], GELU_DEGREES[i], cfg["softgelu"][act])
        print(f"[fit] layer {i}: delta2 {cfg['softmax'][name]['delta2']} counts {cfg['softmax'][name]['recip_iters_cheb']} "
              f"GELU X {cfg['softgelu'][act]['xmax']:.1f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)


if __name__ == "__main__":
    main()
