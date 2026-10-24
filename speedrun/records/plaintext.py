#!/usr/bin/env python3
"""Time unencrypted GPT-2 small at batch 1 on both tracks: the plaintext bar in the records chart.

    CUDA_VISIBLE_DEVICES=0 python3 records/plaintext.py      # writes records/plaintext.json

The same tasks as the encrypted tracks, on the dev prompts. Prefill-128: a forward over 128 tokens and the argmax
of the last position. Decode-8: from a cache of 8 positions, steps at positions 8-15, teacher-forced, each step's
argmax read back to the host before the next starts. Each token id starts on the host, inside the clock. Four ways
to run it: HuggingFace eager in float32, and a fixed-shape forward replayed as CUDA graphs in float32, float16 and
bfloat16. Each is checked with the harness's float64 reference and KL; the bar is the fastest that passes the
tracks' KL bars. Needs torch, safetensors and transformers, and HF_HOME holding openai-community/gpt2.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
from gpt2_ref import GPT2 as Ref, kl  # noqa: E402

D, NH, HD, L, T, FILL = 768, 12, 64, 12, 128, 8
BARS = {"prefill": 1e-2, "decode": 1e-3}
REPEATS = 3


class Static:
    """GPT-2 small on a preallocated KV cache: every shape is fixed, so each step can be captured as a CUDA graph."""

    def __init__(self, sd, dtype):
        g = {k: v.to("cuda", dtype) for k, v in sd.items() if not k.endswith((".attn.bias", ".attn.masked_bias"))}
        self.w, self.blocks = g, [{n[len(f"h.{i}."):]: v for n, v in g.items() if n.startswith(f"h.{i}.")}
                                  for i in range(L)]
        self.k = torch.zeros(L, NH, T, HD, device="cuda", dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def forward(self, ids, pos):
        """Logits of the last of the len(ids) tokens at positions pos, pos + 1, ...; writes their K and V."""
        n = ids.shape[0]
        assert n == 1 or pos == 0
        x = self.w["wte.weight"][ids] + self.w["wpe.weight"][pos:pos + n]
        for i, b in enumerate(self.blocks):
            h = F.layer_norm(x, (D,), b["ln_1.weight"], b["ln_1.bias"], 1e-5)
            q, k, v = (h @ b["attn.c_attn.weight"] + b["attn.c_attn.bias"]).view(n, 3, NH, HD).permute(1, 2, 0, 3)
            self.k[i, :, pos:pos + n], self.v[i, :, pos:pos + n] = k, v
            a = F.scaled_dot_product_attention(q, self.k[i, :, :pos + n], self.v[i, :, :pos + n], is_causal=n > 1)
            x = x + a.transpose(0, 1).reshape(n, D) @ b["attn.c_proj.weight"] + b["attn.c_proj.bias"]
            h = F.layer_norm(x, (D,), b["ln_2.weight"], b["ln_2.bias"], 1e-5)
            h = F.gelu(h @ b["mlp.c_fc.weight"] + b["mlp.c_fc.bias"], approximate="tanh")
            x = x + h @ b["mlp.c_proj.weight"] + b["mlp.c_proj.bias"]
        return F.layer_norm(x[-1], (D,), self.w["ln_f.weight"], self.w["ln_f.bias"], 1e-5) @ self.w["wte.weight"].T


class Graphs:
    """One CUDA graph for the prefill and one per decode position, each ending in the argmax."""

    def __init__(self, sd, dtype):
        self.m = Static(sd, dtype)
        self.ids, self.tok = torch.zeros(T, dtype=torch.long, device="cuda"), torch.zeros(1, dtype=torch.long,
                                                                                            device="cuda")
        self.graphs, self.out = {}, {}
        for key, run in [("prefill", lambda: self.m.forward(self.ids, 0))] + \
                        [(p, lambda p=p: self.m.forward(self.tok, p)) for p in range(FILL, 2 * FILL)]:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                logits = run()
                self.out[key] = (logits, logits.argmax())
            self.graphs[key] = g

    def prefill(self, ids):
        self.ids.copy_(torch.tensor(ids))
        self.graphs["prefill"].replay()
        return self.out["prefill"][1].item(), self.out["prefill"][0]

    def fill(self, ids):
        self.m.forward(torch.tensor(ids, device="cuda"), 0)

    def step(self, tok, pos):
        self.tok.copy_(torch.tensor([tok]))
        self.graphs[pos].replay()
        return self.out[pos][1].item(), self.out[pos][0]


class Eager:
    """HuggingFace's GPT2LMHeadModel as shipped, with its own KV cache."""

    def __init__(self, path, dtype):
        from transformers import GPT2LMHeadModel
        self.m = GPT2LMHeadModel.from_pretrained(path, dtype=dtype).cuda().eval()

    def prefill(self, ids):
        logits = self.m(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1]
        return logits.argmax().item(), logits

    def fill(self, ids):
        self.cache = self.m(input_ids=torch.tensor([ids], device="cuda"), use_cache=True).past_key_values

    def step(self, tok, pos):
        out = self.m(input_ids=torch.tensor([[tok]], device="cuda"), past_key_values=self.cache, use_cache=True)
        self.cache, logits = out.past_key_values, out.logits[0, -1]
        return logits.argmax().item(), logits


def clock(fn):
    t = time.perf_counter()
    r = fn()
    return time.perf_counter() - t, r


def measure(model, prompts, want):
    """Times per prefill and per decode step over REPEATS passes; the KL and argmax checks on the first pass."""
    pre, dec, kl_pre, kl_dec, agree = [], [], [], [], []
    for rep in range(REPEATS + 1):            # pass 0 warms up and checks, the others are timed
        for ids, (ref_pre, ref_dec) in zip(prompts, want):
            s, (tok, logits) = clock(lambda: model.prefill(ids[:T]))
            if rep:
                pre.append(s)
            else:
                kl_pre.append(kl(ref_pre, logits.double().cpu().numpy()))
                agree.append(tok == int(np.argmax(ref_pre)))
            model.fill(ids[:FILL])
            for p in range(FILL, 2 * FILL):
                s, (tok, logits) = clock(lambda: model.step(ids[p], p))
                if rep:
                    dec.append(s)
                else:
                    kl_dec.append(kl(ref_dec[p - FILL], logits.double().cpu().numpy()))
                    agree.append(tok == int(np.argmax(ref_dec[p - FILL])))
    stats = lambda xs: {"mean": statistics.fmean(xs), "median": statistics.median(xs), "n": len(xs)}
    return {"prefill_s": stats(pre), "decode_s_per_token": stats(dec),
            "decode_tokens_per_s": 1 / statistics.fmean(dec),
            "kl_prefill_mean": statistics.fmean(kl_pre), "kl_decode_mean": statistics.fmean(kl_dec),
            "argmax_agreement": sum(agree) / len(agree)}


def main():
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    path = sorted(glob.glob(os.path.join(hub, "models--openai-community--gpt2/snapshots/*")))[-1]
    with open(os.path.join(HERE, "..", "prompts", "dev.json"), encoding="utf-8") as f:
        prompts = json.load(f)["prompts"]
    ref = Ref(os.path.join(path, "model.safetensors"))
    want = []
    for ids in prompts:
        lg = ref.logits(ref.embed(ids[:T]))
        want.append((lg[T - 1], lg[FILL:2 * FILL]))
    torch.set_float32_matmul_precision("highest")
    sd = load_file(os.path.join(path, "model.safetensors"))
    runs = {}
    with torch.inference_mode():
        for name, make in [("hf-eager-fp32", lambda: Eager(path, torch.float32)),
                           ("graphs-fp32", lambda: Graphs(sd, torch.float32)),
                           ("graphs-fp16", lambda: Graphs(sd, torch.float16)),
                           ("graphs-bf16", lambda: Graphs(sd, torch.bfloat16))]:
            runs[name] = measure(make(), prompts, want)
            r = runs[name]
            print(f"{name:14} prefill {r['prefill_s']['mean'] * 1e3:7.3f} ms  KL {r['kl_prefill_mean']:.1e}   "
                  f"decode {r['decode_s_per_token']['mean'] * 1e3:7.3f} ms/token ({r['decode_tokens_per_s']:,.0f} "
                  f"tokens/s)  KL {r['kl_decode_mean']:.1e}   argmax {r['argmax_agreement']:.1%}", flush=True)
            torch.cuda.empty_cache()
    bar = {}
    for track, key in [("prefill", "prefill_s"), ("decode", "decode_s_per_token")]:
        ok = {n: r for n, r in runs.items() if r[f"kl_{track}_mean"] < BARS[track]}
        best = min(ok, key=lambda n: ok[n][key]["mean"])
        bar[track] = {"config": best, "value": ok[best][key]["mean"]}
        print(f"bar {track}: {best}, {bar[track]['value']:.6g} s")
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "date": time.strftime("%Y-%m-%d"), "prompts": "prompts/dev.json", "repeats": REPEATS}
    with open(os.path.join(HERE, "plaintext.json"), "w", encoding="utf-8") as f:
        json.dump({"env": env, "bar": bar, "runs": runs}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
