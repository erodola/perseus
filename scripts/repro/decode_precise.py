"""run_decode with millisecond per-token times (run_decode prints decode= / argmax= to 0.1 s): an import hook wraps
Gpt2Model.run_decode's on_token when run_decode imports the model, so module-level env switches still see the plan's
runtime= settings. Every token prints `[precise] tok<p> decode=<s> argmax=<s> e2e=<s>`, and the steady-token (pos >= 1)
e2e median / mean / range at exit. Same args as run_decode."""
import atexit
import importlib.abc
import importlib.machinery
import os
import runpy
import statistics
import sys

sys.path.insert(0, os.getcwd())
rows = []


def patch(M):
    orig = M.Gpt2Model.run_decode

    def run_decode(self, inputs, gt_logits=None, on_token=None, argmax=False, first=0):
        def tok(rec):
            d, a = rec.get("decode_s", 0.0), rec.get("argmax_s", 0.0)
            rows.append((rec.get("pos"), d, a))
            print(f"[precise] tok{rec.get('pos')} decode={d:.3f} argmax={a:.3f} e2e={d + a:.3f}", flush=True)
            if on_token:
                on_token(rec)
        return orig(self, inputs, gt_logits, on_token=tok, argmax=argmax, first=first)
    M.Gpt2Model.run_decode = run_decode


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "examples.gpt2_from_primitives.model":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module

        def exec_module(mod, _orig=orig):
            _orig(mod)
            patch(mod)
        spec.loader.exec_module = exec_module
        return spec


@atexit.register
def summary():
    st = [r for r in rows if r[0] is not None and r[0] >= 1]
    if st:
        e = [d + a for _, d, a in st]
        print(f"[precise] steady n={len(st)} e2e median={statistics.median(e):.3f} mean={statistics.mean(e):.3f} "
              f"min={min(e):.3f} max={max(e):.3f} | decode median={statistics.median([d for _, d, _ in st]):.3f} "
              f"argmax median={statistics.median([a for _, _, a in st]):.3f}", flush=True)


sys.meta_path.insert(0, Hook())
sys.argv = ["run_decode"] + sys.argv[1:]
runpy.run_module("examples.gpt2_from_primitives.run_decode", run_name="__main__", alter_sys=True)
