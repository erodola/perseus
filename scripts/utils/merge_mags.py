#!/usr/bin/env python
"""Max-merge the magnitudes of captures taken at later positions into a position-0 capture, by ciphertext name.

A plan sizes every refresh (its correction factor) from the magnitudes its capture recorded, and a decode capture
records one position. Some values grow with the cache: GPT-2's softmax refresh inputs reach ~800x their position-0
magnitude by 511 keys, past the wrap of a factor sized at position 0. With step-scoped names (IMPL_SCOPED_NAMES=1)
every op outside a cache-shaped step keeps its name at every position, so the position-0 graph's nodes find their
later twins by name and take the largest output_max_* over the captures given. Nodes without a twin (the q.K^T,
whose scope carries the cache shape) keep their own statistics.

    # the later position on the numpy fake (FHE_GRAPH_CAPTURE_TOKEN selects it), with the capture's own --set switches
    FHE_GRAPH_CAPTURE_TOKEN=511 .venv/bin/python -m examples.gpt2_from_primitives.run_decode --fake --tokens 512 \
        --capture /tmp/pos511 --configs <configs> --set IMPL_SCOPED_NAMES=1 ...
    .venv/bin/python scripts/utils/merge_mags.py graphs/<pos0> graphs/<merged> /tmp/pos511
"""
import argparse
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="the position-0 capture (block_<b>/graph.json)")
    ap.add_argument("out", help="the merged capture (a new directory)")
    ap.add_argument("extra", nargs="+", help="captures of the same model at later positions")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")
    shutil.copytree(a.base, a.out)
    for blk in sorted(d for d in os.listdir(a.base) if d.startswith("block_")):
        path = os.path.join(a.out, blk, "graph.json")
        with open(path) as f:
            g = json.load(f)
        twins = []
        for e in a.extra:
            p = os.path.join(e, blk, "graph.json")
            if os.path.exists(p):
                with open(p) as f:
                    twins.append({n["output"]: n for n in json.load(f)["nodes"]})
        matched = grown = 0
        top = []
        for n in g["nodes"]:
            for t in twins:
                m = t.get(n["output"])
                if m is None or m.get("op_type") != n.get("op_type"):
                    continue
                matched += 1
                ratio = (m.get("output_max_abs") or 0.0) / max(n.get("output_max_abs") or 0.0, 1e-30)
                for k, v in m.items():
                    if k.startswith("output_max_") and isinstance(v, (int, float)) and v > (n.get(k) or 0.0):
                        n[k] = v
                if ratio > 1.0:
                    grown += 1
                    top.append((ratio, n["output"]))
        with open(path, "w") as f:
            json.dump(g, f)
        top.sort(reverse=True)
        print(f"{blk}: {len(g['nodes'])} nodes, {matched} matched, {grown} grew"
              + ("; largest " + ", ".join(f"{v} x{r:.0f}" for r, v in top[:3]) if top else ""), flush=True)


if __name__ == "__main__":
    main()
