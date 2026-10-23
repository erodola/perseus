"""Planner-equivalence vectors: run the OLD planner sim (perseus/plan/placer/sim.py::simulate) on a shipped plan.

Usage (read-only on the old repo; no bytecode is written there):
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data02/users/mithrillo/projects/perseus \
        /data02/users/mithrillo/projects/perseus/.venv/bin/python crates/pc-shadow/tests/tools/sim_dump.py \
        <graph dir> <placement dir> <out.json> [KEY=VALUE ...]

KEY=VALUE pairs are planner env the plan was made with beyond its recipe= line (test_paper_plans.py GATE extras).
Run from the repository root with relative dirs: the dump records the paths as given.

The final sim of a plan is reconstructed from the shipped placement file (`dump_block`: refreshed landings, hint_force,
realized anchors, chained seeds) and must reproduce its `final_named_levels/degs` for every output, or is refused.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path


def recipe_env(placement_dir: Path) -> dict[str, str]:
    """The PLAN_* knobs of the recipe line of PLAN_CMD.txt that the sim / graph loader read."""
    env: dict[str, str] = {}
    p = placement_dir / "PLAN_CMD.txt"
    if not p.is_file():
        return env
    for line in p.read_text().splitlines():
        if not line.startswith("recipe="):
            continue
        for tok in line[len("recipe="):].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                env[k] = v       # later tokens win, as in the shell
    return env


SIM_ENV = ("PLAN_DELIB_LANDING_REMAP", "PLAN_DELIB_DEG", "PLAN_STC_FIRST_KV_LANDING",
           "PLAN_STC_FIRST_ENTRY", "PLAN_STC_FIRST_FALLBACK", "PLAN_DELIBERATE_CLAMP0")


def main() -> None:
    graph_dir, placement_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    env = recipe_env(placement_dir)
    env.update(kv.split("=", 1) for kv in sys.argv[4:])
    for k in SIM_ENV:
        os.environ.pop(k, None)
        if k in env:
            os.environ[k] = env[k]
    # the modules read the env at import time (sim.py _DELIB_DEG_CAPTURED)
    from perseus.plan.placer.ir import Graph, parse_lvl_suffix
    from perseus.plan.placer.place import erase_reactive_bootstraps
    from perseus.plan.placer.sim import Budget, simulate
    from perseus.plan.placer.planner import fed_back_entry

    unit = int(env.get("PLAN_LEVEL_UNIT", "2"))
    blocks = sorted((d for d in graph_dir.glob("block_*") if (d / "graph.json").is_file()),
                    key=lambda d: int(d.name.rsplit("_", 1)[1]))
    out_blocks = []
    entry: tuple[int, int] | None = None
    if env.get("FIRST_ENTRY_LEVEL"):
        entry = (int(env["FIRST_ENTRY_LEVEL"]), int(env.get("FIRST_ENTRY_DEG", "1")))
    plans = [(bd, placement_dir / f"{bd.name}_placement.json") for bd in blocks]
    fb = placement_dir / "block_0_feedback_placement.json"
    if fb.is_file():
        plans.append((graph_dir / "block_0", fb))
    for bd, pf in plans:
        if not pf.is_file():
            print(f"skip {bd.name}: no placement", file=sys.stderr)
            continue
        plan = json.loads(pf.read_text())
        feedback = pf.name.endswith("_feedback_placement.json")
        blk_entry = fed_back_entry(bd / "graph.json") if feedback else entry
        rec = dump_block(bd, pf, plan, unit, blk_entry, Graph, erase_reactive_bootstraps, Budget, simulate,
                         parse_lvl_suffix)
        out_blocks.append(rec)
        if not feedback:
            s = plan["summary"]
            entry = (int(s["exit_level"]), int(s.get("exit_deg") or 1)) if s.get("exit_level") is not None else None
        print(f"{bd.name}{' (feedback)' if feedback else ''}: {rec['n_nodes']} nodes, "
              f"{rec['summary_vars']} named vars == summary", file=sys.stderr)
    doc = {
        "graph_dir": str(graph_dir),
        "placement_dir": str(placement_dir),
        "level_unit": unit,
        "env": {k: env[k] for k in SIM_ENV if k in env},
        "blocks": out_blocks,
    }
    out_path.write_text(json.dumps(doc, separators=(",", ":")))


def dump_block(bd, pf, plan, unit, entry, Graph, erase, Budget, simulate, parse_lvl_suffix) -> dict:
    raw = json.loads((bd / "graph.json").read_text())
    g0 = Graph.from_nodes(raw["nodes"], level_unit=unit)
    g, _ = erase(g0)
    rules = plan["rules"]
    bl = float(rules["bootstrap_level"])
    max_level = float(rules["max_level"])
    source_level = float(rules["source_level"])
    cache_read_level = float(rules["cache_read_level"])
    entry_level, entry_deg = (entry if entry is not None else (None, None))

    # plan_block's seeds, verbatim
    input_level_of: dict[str, float] = {}
    first_consumer_seed: dict[str, float] = {}
    for n in g.nodes:
        for i, v in enumerate(n.inputs):
            if v not in g.inputs:
                continue
            if v not in input_level_of and i < len(n.input_levels) and n.input_levels[i] is not None:
                input_level_of[v] = float(n.input_levels[i])
            if v not in first_consumer_seed and n.output_level is not None:
                first_consumer_seed[v] = float(n.output_level) - n.cost
    entry_var = next((v for n in g.nodes for v in n.inputs if v in g.inputs), None)

    def is_cache_side_input(v):
        return entry_level is not None and v != entry_var and v.startswith("anon_") and v in input_level_of

    def seed_consumed(v):
        exp = parse_lvl_suffix(v)
        if "cache." in v:
            lv = exp if exp is not None else cache_read_level
            return lv - bl
        if is_cache_side_input(v):
            return input_level_of[v] - bl
        if entry_level is not None:
            return entry_level - bl
        if v in first_consumer_seed:
            return first_consumer_seed[v] - bl
        if exp is not None:
            return exp - bl
        if v in input_level_of:
            return input_level_of[v] - bl
        return source_level - bl

    def seed_deg(v):
        if "cache." in v or is_cache_side_input(v):
            return 2
        if entry_deg is not None:
            return entry_deg
        exp = parse_lvl_suffix(v)
        fc = first_consumer_seed.get(v)
        if exp is not None and fc is not None:
            return 2 if exp > fc else 1
        return 2 if seed_consumed(v) > 0 else 1

    s = plan["summary"]
    fnl, fnd = s["final_named_levels"], s.get("final_named_degs", {})
    placed = [p["target_var"] for p in plan["placements"]]
    hint_fire = set(plan.get("hint_fire", []))
    sparse = set((plan.get("sparse_slots") or {}).keys())
    refreshed: dict[str, tuple[float, int]] = {}
    for v in list(placed) + sorted(hint_fire):
        refreshed[v] = (float(fnl[v]) - bl, int(fnd.get(v, 1)))
    hint_force = {n.output: (n.output in hint_fire) for n in g.nodes if n.hint_level is not None and n.output}
    quality = set(placed) | hint_fire
    # a boundary realize of the exit var (emit, after the sim) is listed here too and has the same effect
    realized = {v for v in plan.get("rescale_after", []) if v not in quality}
    dense_fallback = None
    e, f = os.environ.get("PLAN_STC_FIRST_ENTRY"), os.environ.get("PLAN_STC_FIRST_FALLBACK")
    if e and f:
        lv, _, dg = f.partition(":")
        dense_fallback = (float(e), float(lv) - bl, int(dg or 1))
    clamp0 = os.environ.get("PLAN_DELIBERATE_CLAMP0", "0") not in ("", "0")
    budget = Budget(L=max_level - bl, unit=unit)
    sim = simulate(g, bootstrap_level=bl, budget=budget, seed_consumed=seed_consumed, seed_deg=seed_deg,
                   refreshed=refreshed, sparse_refreshed=sparse, placed=set(placed), realized=realized,
                   hint_force=hint_force, deliberate_clamp0=clamp0, dense_fallback=dense_fallback)

    # the reconstruction check: the planner's own final sim is what wrote final_named_levels/degs
    bad = []
    for n in g.nodes:
        if not n.output:
            continue
        lv = int(bl + math.ceil(sim.consumed.get(n.output, 0.0)))
        dg = int(sim.deg.get(n.output, 1))
        if lv != fnl.get(n.output) or (fnd and dg != fnd.get(n.output)):
            bad.append((n.output, lv, dg, fnl.get(n.output), fnd.get(n.output)))
    if bad:
        raise SystemExit(f"{pf}: reconstruction differs from the plan summary on {len(bad)} vars: {bad[:5]}")

    # a second, plan-free run: no refreshes, hints decided by their thresholds (fired ones land at the sim's
    # default (0, 2)), so the deep levels, the hint rule and the budget guard are exercised too
    bare = simulate(g, bootstrap_level=bl, budget=budget, seed_consumed=seed_consumed, seed_deg=seed_deg,
                    refreshed={}, deliberate_clamp0=clamp0, dense_fallback=dense_fallback)

    return {
        "block": bd.name,
        "graph_file": str(bd / "graph.json"),
        "placement_file": str(pf),
        "bootstrap_level": bl,
        "budget_L": max_level - bl,
        "entry": list(entry) if entry is not None else None,
        "seeds": {v: [seed_consumed(v), seed_deg(v)] for v in sorted(g.inputs)},
        "refreshed": {v: [c, d] for v, (c, d) in refreshed.items()},
        "sparse_refreshed": sorted(sparse),
        "placed": sorted(placed),
        "realized": sorted(realized),
        "hint_force": hint_force,
        "deliberate_clamp0": clamp0,
        "dense_fallback": list(dense_fallback) if dense_fallback else None,
        "n_nodes": len(g.nodes),
        "summary_vars": sum(1 for n in g.nodes if n.output),
        "outputs": [n.output for n in g.nodes],
        "planned": expect(g, sim),
        "bare": expect(g, bare),
    }


def expect(g, sim) -> dict:
    """Per node (topological order): the rule's output before any override, and the var's final state."""
    return {
        "node_out": [sim.node_out.get(n.idx) for n in g.nodes],
        "node_deg": [sim.node_deg.get(n.idx) for n in g.nodes],
        "final_c": [sim.consumed.get(n.output) if n.output else None for n in g.nodes],
        "final_d": [sim.deg.get(n.output) if n.output else None for n in g.nodes],
        "hint_fired": {str(k): v for k, v in sorted(sim.hint_fired.items())},
        "over_budget": [[i, a] for i, a in sim.over_budget],
    }

if __name__ == "__main__":
    main()
