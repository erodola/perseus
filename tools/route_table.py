#!/usr/bin/env python3
"""Write the planner's route table (`PLAN_ACC_TABLE=<out>`): the runtime's measured refresh costs, landings and
precision over the base accuracy table, which stays the planner's default.

    python3 tools/route_table.py --bench <pc-bench bts --all --keys packed log> [--acc docs/acc_n32.json] [--out ..]
    python3 tools/route_table.py --recipe <plan dir> [--stcf]     # the landing knobs it means for that plan's runtime=

Adds to the base: the measured sparse curves (SparseB-512/256, SPRU), `bts_ms` / `bts_ms_k` per route, `landings`
(realized, drop 0); with --stcf-curves also `stcf` curves by slot amplitude + the floor F of `bend(a, cf) + F * 2^cf
/ a` per read (the planner then prices dense StC-first sites on them: off by default, their max-error optimum picks
CFs that cost in-model accuracy). Shipped-order dense curves stay the base's (period-1 law).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "python" / "perseus" / "plan" / "data"
# acc_n32.json route name -> (planner route key, period of the measured payload)
SPARSE = {"sparse512": (512, 512), "sparse256": (256, 256), "spru_h64": (1, 1)}
BENCH_ROUTES = ("dense", "stcf", "512", "256", "1")


def bend(a: float, cf: int) -> float:
    x = 2.0 * math.pi * a / 2.0 ** cf
    return abs(1.0 - math.sin(x) / x) if x > 1e-12 else 0.0


def bench_ms(log: Path) -> dict[str, dict[int, float]]:
    """{route: {K: eager ms per ciphertext}} from the `pc-bench bts --all` route x K table."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for line in log.read_text().splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", line)
        if m and m.group(1) in BENCH_ROUTES:
            out[m.group(1)][int(m.group(2))] = float(m.group(3))
    missing = [r for r in BENCH_ROUTES if 1 not in out.get(r, {})]
    if missing:
        raise SystemExit(f"{log}: no K=1 row for route(s) {missing} (run pc-bench bts --all --k 1,...)")
    return dict(out)


def recipe(table: dict, plan_dir: Path, stcf: bool, unit: int = 2) -> str:
    """The planner landing knobs the table's measured landings mean under `plan_dir`'s runtime= line."""
    rt = {}
    for line in (plan_dir / "PLAN_CMD.txt").read_text().splitlines():
        if line.startswith("runtime="):
            rt = dict(kv.split("=", 1) for kv in line[len("runtime="):].split() if "=" in kv)
    drop = {int(s): int(k) for s, k in (x.split(":") for x in rt.get("FIDESLIB_BTS_RAISE_DROP", "").split(",") if x)}
    land = table["landings"]
    at = lambda r: land[r] + unit * drop.get(int(r) if r.isdigit() else 0, 0)  # noqa: E731
    sparse = [s for s in rt.get("SPARSE_BTS_SLOTS", "512,1").split(",") if s and s != "0"]
    out = [f"BTS_LEVEL={at('stcf') if stcf else at('dense')}",
           "PLAN_SPARSE_BTS_OUT=" + ",".join(f"{s}:{at(s)}" for s in sorted(sparse, key=int) if s in land)]
    if stcf:
        out.append(f"PLAN_STC_FIRST_FALLBACK={at('dense')}:1")
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", type=Path, help="log of pc-bench bts --all --keys packed")
    ap.add_argument("--recipe", type=Path, help="print the landing knobs for this plan dir (reads --out)")
    ap.add_argument("--stcf", action="store_true", help="--recipe: the dense route runs StC-first")
    ap.add_argument("--stcf-curves", action="store_true", help="write the StC-first precision curves (planner pricing)")
    ap.add_argument("--acc", type=Path, default=ROOT / "docs" / "acc_n32.json")
    ap.add_argument("--base", type=Path, default=DATA / "bts_accuracy_n32.json")
    ap.add_argument("--out", type=Path, default=DATA / "route_table_n32.json")
    a = ap.parse_args()
    if a.recipe:
        print(recipe(json.loads(a.out.read_text()), a.recipe, a.stcf))
        return
    if a.bench is None:
        ap.error("--bench is required to write the table")

    base = json.loads(a.base.read_text())
    acc = json.loads(a.acc.read_text())
    ms = bench_ms(a.bench)

    landings: dict[str, int] = {}
    names = {"dense": "dense", "stc_first": "stcf", "sparse512": "512", "sparse256": "256", "spru_h64": "1"}
    for r in acc["rows"]:
        # the dense orders' rows encrypted at a plan's entry (44); the sparse routes' at their usual entry
        dense = r["route"] in ("dense", "stc_first")
        if r["route"] in names and r.get("landing_level") is not None and (not dense or r.get("input_level") == 44):
            lv = int(r["landing_level"]) - 2 * int(r.get("raise_drop") or 0)
            key = names[r["route"]]
            if landings.setdefault(key, lv) != lv:
                raise SystemExit(f"{a.acc}: route {key} lands at {landings[key]} and {lv}")
    sparse: dict[tuple[int, int, int], dict[float, float]] = defaultdict(dict)
    stcf: dict[tuple[int, bool], dict[float, float]] = defaultdict(dict)
    for r in acc["rows"]:
        if r.get("rel_err") is None:
            continue
        cf, amp, e = int(r["cf"]), float(r["amp"]), float(r["rel_err"])
        if r["route"] in SPARSE and not r.get("real") and int(r.get("period") or 0) == SPARSE[r["route"]][1]:
            route, period = SPARSE[r["route"]]
            d = sparse[(cf, period, route)]
            d[amp] = max(d.get(amp, 0.0), e)
        elif r["route"] == "stc_first" and r.get("period") == 0:
            d = stcf[(cf, bool(r.get("real")))]
            d[amp] = max(d.get(amp, 0.0), e)

    wall = [w for w in base["wall"] if (int(w["cf"]), int(w["period"]), int(w["route"])) not in sparse]
    for (cf, period, route), d in sorted(sparse.items()):
        wall += [{"cf": cf, "period": period, "route": route, "amp": amp, "rel_err": e} for amp, e in sorted(d.items())]

    floor: dict[str, float] = {}
    for real in (False, True):
        fs = [(e - bend(amp, cf)) * amp / 2.0 ** cf for (cf, rr), d in stcf.items() if rr == real
              for amp, e in d.items() if e - bend(amp, cf) > 0.5 * e]
        floor["real" if real else "complex"] = max(fs)
    fit = []
    for (cf, real), d in sorted(stcf.items()):
        f = floor["real" if real else "complex"]
        for amp, e in sorted(d.items()):
            fit.append(abs(math.log10((bend(amp, cf) + f * 2.0 ** cf / amp) / e)))
    fit.sort()

    doc = dict(base)
    doc["wall"] = wall
    if a.stcf_curves:
        doc["stcf"] = [{"cf": cf, "real": real, "amp": amp, "rel_err": e}
                       for (cf, real), d in sorted(stcf.items()) for amp, e in sorted(d.items())]
        doc["stcf_floor"] = floor
    doc["landings"] = landings
    doc["bts_ms"] = {r: ms[r][1] for r in BENCH_ROUTES}
    doc["bts_ms_k"] = {r: {str(k): v for k, v in sorted(ms[r].items())} for r in BENCH_ROUTES}
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()  # noqa: E731
    doc["meta"] = dict(base.get("meta", {}), route_table={
        "base": f"{a.base.name} (md5 {md5(a.base)})", "acc": f"{a.acc.name} (md5 {md5(a.acc)})",
        "bench": a.bench.name, "replaced": sorted(f"cf{k[0]}_p{k[1]}_r{k[2]}" for k in sparse),
        "stcf_fit_abs_log10_ratio": {"median": fit[len(fit) // 2], "max": fit[-1], "cells": len(fit)}})
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[route_table] {a.out}: {len(sparse)} sparse curves replaced, {len(doc.get('stcf', []))} StC-first cells, "
          f"floor {floor}, law vs measured |log10| median {fit[len(fit) // 2]:.3f} max {fit[-1]:.3f}")
    print("[route_table] ms/ct K=1: " + ", ".join(f"{r} {v:.2f}" for r, v in doc["bts_ms"].items())
          + "; landings " + ", ".join(f"{r} {v}" for r, v in landings.items()))


if __name__ == "__main__":
    main()
