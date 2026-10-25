"""Every shipped plan regenerates from its graph with the recipe in its PLAN_CMD.txt.

Plans made with ``perseus/examples/gpt2/make_plan.sh`` are regenerated with it (blocks, argmax stage, feedback
stage). The main plan is always checked; PERSEUS_ALL_PLANS=1 adds the baselines and ablations, PERSEUS_PLANS=a,b
names plans. Comparison: `PLAN_FIELDS` field by field, then the runtime body (perseus.plan.equiv) with the
contract stamps left out. The planner gate (`PC_PLAN_GATE=1`; by hand: `python python/tests/test_paper_plans.py
[name ...]`) replans `GATE`, whose recipes spell out what incomplete PLAN_CMD.txt files omit. DaCapo / Orion
upstream-tool plans skip (their tools are not ported).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from perseus._paths import ASSETS, PKG
from perseus.plan import contract as _contract
from perseus.plan import equiv as _equiv

REPO = ASSETS
BP = REPO / "bootstrap_placements"
MAIN = "gpt2_decode_python_n32"
# the released DaCapo / Orion drivers were repo scripts, not ported
UPSTREAM = ("scripts/utils/orion_upstream/plan.sh", "scripts/utils/dacapo_upstream/plan.sh")
GPT2_TOOL = "python/perseus/examples/gpt2/make_plan.sh"
LLAMA_TOOL = "python/perseus/examples/llama/make_plan.sh"
# tool spellings in PLAN_CMD.txt (the C++-runtime layout, then this one) -> the package's script
TOOLS = {
    "examples/gpt2_from_primitives/make_plan.sh": PKG / "examples" / "gpt2" / "make_plan.sh",
    "examples/llama_from_primitives/make_plan.sh": PKG / "examples" / "llama" / "make_plan.sh",
    GPT2_TOOL: PKG / "examples" / "gpt2" / "make_plan.sh",
    LLAMA_TOOL: PKG / "examples" / "llama" / "make_plan.sh",
}
LLAMA_TOOLS = (LLAMA_TOOL, "examples/llama_from_primitives/make_plan.sh")
# the stamp keys: a contract, not plan content
STAMPS = ("capture_env", "config")
# what the runtime binds of a plan, compared one by one (`summary.*` = nested summary keys)
PLAN_FIELDS = ("placements", "rescale_after", "sparse_slots", "correction_factor", "real_route", "raise_drop",
               "hint_fire", "summary.final_named_levels", "summary.final_named_degs")

# The StC-first prefill pipes' extra planner env (old repo logs/prefill/q_g2stcf.sh / q_stcf.sh `K`): in no
# PLAN_CMD.txt (gpt2's has no recipe= line; llama's records only make_plan.sh's own RECIPE keys).
_STCF = ("BTS_LEVEL=28 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_LANDING_MAX=42 PLAN_RAISE_ROUTES=0 "
         "PLAN_REFRESH_ENV_CAP_ROUTES=0:46 PLAN_SLIM_CUBIC=1 PLAN_STC_FIRST_ENTRY=46 PLAN_STC_FIRST_FALLBACK=34:1")
# the route table's measured landings (`tools/route_table.py --recipe`, the blocked prefill / encoder runtime=)
_M16L = ("BTS_LEVEL=34 PLAN_SPARSE_BTS_OUT=1:16,512:24 "
         f"PLAN_ACC_TABLE={PKG / 'plan' / 'data' / 'route_table_n32.json'}")
_M16S = "PLAN_SPARSE_BTS_OUT=1:16,512:24 " + _STCF
# StC-first captures whose deliberate refreshes past the entry take the shipped order (no StC-first input cap)
_M18S = _M16S.replace("PLAN_REFRESH_ENV_CAP_ROUTES=0:46 ", "") + " PLAN_STC_FIRST_REROUTE=1"
# name -> (graph override, recipe beyond PLAN_CMD.txt's recipe= line, tool override, stages); None = as shipped
GATE: dict[str, tuple | None] = {
    "gpt2_decode_python_n32": None,
    # ILP placer (scipy milp = HiGHS); PLAN_CMD.txt has no tool= line but the plan ships argmax + feedback stages.
    # Shipped 2026-10-07 (old repo ce7ac6a), before make_plan.sh defaulted the real route + raise-drop set (d5db400,
    # 10-08) and shared realize anchors (6d23015, 10-12): those defaults off, as the recipe then ran
    "gpt2_decode_python_n32_ilp": (None, "PLAN_REAL_ROUTE=0 PLAN_RAISE_DROP_SET= PLAN_SHARED_REALIZE=", GPT2_TOOL,
                                   None),
    # logs/prefill/pipe2.sh: gpt2's plan_env is empty
    "gpt2_prefill_python_n32_pcmmstcf": ("graphs/gpt2_prefill_python_n32_pcmmstcf", _STCF,
                                         GPT2_TOOL, ("blocks",)),
    # shipped 2026-10-11 (9a65ce9), before make_plan.sh defaulted shared realize anchors (6d23015, 10-12)
    "llama31_8b_decode_python_n32": (None, "PLAN_SHARED_REALIZE=", None, None),
    # the hand-off decode plan (old repo logs/prefill/HANDOFF.md, 2026-10-14), as its recipe= line ran
    "llama31_8b_decode_python_n32_dsw48": None,
    # dsw48 cut for a prefill hand-off's cache (KV_ENTRY_LEVEL + PLAN_KEEP_PLACEMENTS in the recipe= line): pr9
    # hands K / V at 36, llama31_8b_prefill128 at 38
    "llama31_8b_decode_python_n32_dsw48_ho36": None,
    "llama31_8b_decode_python_n32_dsw48_ho38": None,
    # ho38 with every softmax refresh one CF lower (PLAN_SITE_CF=softmax:-1 in its recipe= line)
    "llama31_8b_decode_python_n32_dsw48_ho38_m18e": None,
    # logs/prefill/pipe.sh llama plan_env (PLAN_DRIFT_RECUT=1 is not in the recipe= line) + the StC-first K
    "llama31_8b_prefill_python_n32_pcmmstcf": (None, "PLAN_DRIFT_RECUT=1 " + _STCF, None, None),
    # the encoder pipes (logs/bert/pipe.sh, logs/vit/pipe_cc.sh) rewrote PLAN_CMD.txt to runtime= / configs= only:
    # gpt2 make_plan.sh blocks stage with PLAN_DELIB_DEG=captured (local_env.sh's only planner input,
    # PLAN_SPARSE_SLOTS=512,1, is make_plan.sh's default)
    "bert_base_python_n32_cc2": ("graphs/bert_base_python_n32_cc2", "PLAN_DELIB_DEG=captured", GPT2_TOOL, ("blocks",)),
    "vit_base_224_python_n32_cc3": ("graphs/vit_base_224_python_n32_cc3", "PLAN_DELIB_DEG=captured", GPT2_TOOL,
                                    ("blocks",)),
    # the paged prefill pipe (logs/fin/q_pr.sh) exported these on top of the recipe= line it recorded
    "llama31_8b_prefill_python_n32_pr2": (None, "PLAN_DELIB_DEG=captured PLAN_DRIFT_RECUT=1", None, None),
    "llama31_8b_prefill_python_n32_pr9": (None, "PLAN_DELIB_DEG=captured PLAN_DRIFT_RECUT=1", None, None),
    # the 2026-10-19 result plans (pers/dev 31ab85f, docs/RESULTS_2026-10-19.md): gpt2_decode_python_n32 is MAIN
    # unchanged; three are renames of plans above (same files) and replan from the same graphs and recipes
    "bert_base_sst2_128": ("graphs/bert_base_sst2_128", "PLAN_DELIB_DEG=captured", GPT2_TOOL, ("blocks",)),
    "vit_b16_224": ("graphs/vit_b16_224", "PLAN_DELIB_DEG=captured", GPT2_TOOL, ("blocks",)),
    # = gpt2_prefill_python_n32_blocked; its graph is a fake v2 capture (0 structural diffs vs the old repo's,
    # which was never tracked), recipe as the blocked plan's REPLAN.sh
    "gpt2_prefill128": ("graphs/gpt2_prefill128", "PLAN_PRUNE=1", GPT2_TOOL, ("blocks",)),
    # pr2's graph + the recipe= line it records, with q_pr.sh's exports (as pr2/pr9)
    "llama31_8b_prefill128": (None, "PLAN_DELIB_DEG=captured PLAN_DRIFT_RECUT=1", None, None),
    # replans on the new runtime's measured route table (tools/route_table.py): landings (`--recipe`), sparse curves;
    # `_m16l` the shipped captures, `_m16s0` captures with StC-first dense refreshes (runtime= adds the switches)
    "gpt2_prefill128_m16l": ("graphs/gpt2_prefill128", "PLAN_PRUNE=1 " + _M16L, GPT2_TOOL, ("blocks",)),
    "bert_base_sst2_128_m16l": ("graphs/bert_base_sst2_128", "PLAN_DELIB_DEG=captured " + _M16L, GPT2_TOOL, ("blocks",)),
    "vit_b16_224_m16l": ("graphs/vit_b16_224", "PLAN_DELIB_DEG=captured " + _M16L, GPT2_TOOL, ("blocks",)),
    "gpt2_prefill128_m16s0": ("graphs/gpt2_prefill128_stcf_m16", "PLAN_PRUNE=1 " + _M16S, GPT2_TOOL, ("blocks",)),
    "bert_base_sst2_128_m16s0": ("graphs/bert_sst2_128_stcf_m16", "PLAN_DELIB_DEG=captured " + _M16S, GPT2_TOOL,
                                 ("blocks",)),
    "vit_b16_224_m18s3": ("graphs/vit_b16_224_stcf_m16", "PLAN_DELIB_DEG=captured " + _M18S, GPT2_TOOL, ("blocks",)),
    # the main recipe (recipe= line) on the ILP placer (objective: total refresh count)
    "gpt2_decode_python_n32_m16i": (None, "PLAN_PLACER=ilp PLAN_ILP_TIME_LIMIT=600", GPT2_TOOL, None),
    # m16i with the shipped order's CF where that order runs (its recipe= line has the whole recipe)
    "gpt2_decode_python_n32_m16i_fcf": None,
    # Decode-8: the ILP placer, one optimum on every machine (PLAN_ILP_TIEBREAK), every refresh raised below the chain
    # top as far as its block allows, the shipped order's CF where it runs, the argmax on the min-cut, on the merged
    # capture of the configs fitted to Decode-8's 16 positions (gpt2_base_n32_kc16; its recipe= and capture= lines); its
    # CutMax variant gpt2_decode_python_n32_kc16_cm comes from speedrun/make_cutmax_plan.sh
    "gpt2_decode_python_n32_kc16": None,
}
# plans this planner no longer regenerates, and why: the gate reports them instead of replanning them
RETIRED = {
    # made by the ILP transcription before its model became the sim's (c064188): 14 of their 15 block files are the
    # min-cut's (fallback:model-mismatch), where the model now solves; they regenerate with that planner (f19831a)
    "gpt2_decode_python_n32_m16i": "made by the ILP's earlier transcription (14/15 block files the min-cut's); it "
                                   "regenerates at f19831a",
    "gpt2_decode_python_n32_m16i_fcf": "made by the ILP's earlier transcription (14/15 block files the min-cut's); "
                                       "it regenerates at f19831a",
}
# tools that capture and plan in one script (their own regeneration check): not replanned here
SELF_CONTAINED = ("speedrun/make_prefill_plan.sh",)
# the 10-19 table plans: `python python/tests/test_paper_plans.py table` replans these five
TABLE = ("llama31_8b_prefill128", "gpt2_prefill128", "gpt2_decode_python_n32", "bert_base_sst2_128", "vit_b16_224")


# summary diagnostics a later planner computes differently than the shipped file recorded (same on the old sim.py and
# on the Shadow-backed sim; no runtime field): left out of the body comparison of that plan only
SUMMARY_DRIFT = {
    # placed_input_levels / _hist were redefined after this plan shipped (2026-10-07)
    "gpt2_decode_python_n32_ilp": ("bts_quality.placed_input_levels", "bts_quality.placed_input_level_hist"),
    # a v1 capture's dense deliberate refresh is marked StC-first when the recipe has an StC-first entry (plan_block):
    # blocks 2 and 9 count one more blind branch (380 -> 381)
    "gpt2_prefill_python_n32_pcmmstcf": ("bts_quality.blind_branches",),
}
# summary lists the planner fills in set-iteration order (PYTHONHASHSEED): compared as sets, every plan
UNORDERED = ("bts_quality.env_cap.placed_past_cap",)


def _plan_dirs():
    if os.environ.get("PERSEUS_PLANS"):     # comma-separated plan names: those only
        return os.environ["PERSEUS_PLANS"].split(",")
    dirs = [MAIN]
    if os.environ.get("PERSEUS_ALL_PLANS") == "1":
        for p in sorted(BP.rglob("PLAN_CMD.txt")):
            rel = str(p.parent.relative_to(BP))
            if rel != MAIN:
                dirs.append(rel)
    return dirs


def _recipe(plan_dir: Path):
    """(graph, recipe flags, tool, route) from PLAN_CMD.txt; None for a measured artifact (no
    `recipe=`). tool is None for the plain run_bootstrap_all_blocks.sh recipe; route is
    `dense` for an upstream tool's dense plan."""
    graph = recipe = tool = None
    route = "sparse"
    for line in (plan_dir / "PLAN_CMD.txt").read_text().splitlines():
        if line.startswith("graph="):
            graph = line[len("graph="):].split(" ")[0]
        elif line.startswith("recipe="):
            recipe = shlex.split(line[len("recipe="):])
        elif line.startswith("tool="):
            tool = line[len("tool="):].split(" ")[0]
        elif line.startswith("route="):
            route = line[len("route="):].strip()
    if recipe is None:
        return None
    assert graph, f"{plan_dir}/PLAN_CMD.txt lacks graph="
    return graph, recipe, tool, route


def _get(doc, field: str):
    for k in field.split("."):
        if not isinstance(doc, dict) or k not in doc:
            return None
        doc = doc[k]
    return doc


def _count(ref, new) -> int:
    """Differing leaves between two values (dict keys / list items; type-strict leaves)."""
    if isinstance(ref, dict) and isinstance(new, dict):
        return sum(_count(ref.get(k), new.get(k)) for k in set(ref) | set(new))
    if isinstance(ref, list) and isinstance(new, list):
        return sum(_count(a, b) for a, b in zip(ref, new)) + abs(len(ref) - len(new))
    return 0 if (type(ref) is type(new) and ref == new) else 1


def field_diffs(new_path: Path, ref_path: Path) -> dict[str, int]:
    """Per PLAN_FIELDS entry, the number of differing leaves (0 = identical, including both absent)."""
    new, ref = (json.loads(Path(p).read_text()) for p in (new_path, ref_path))
    return {f: _count(_get(ref, f), _get(new, f)) for f in PLAN_FIELDS}


def _without_stamps(src: Path, dst: Path, drift: tuple[str, ...] = ()) -> Path:
    doc = json.loads(src.read_text())
    for k in STAMPS:
        doc.pop(k, None)
    for path in drift:
        *up, last = path.split(".")
        d = doc.get("summary", {})
        for k in up:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict):
            d.pop(last, None)
    for path in UNORDERED:
        *up, last = path.split(".")
        d = doc.get("summary", {})
        for k in up:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict) and isinstance(d.get(last), list):
            d[last] = sorted(d[last], key=json.dumps)
    dst.write_text(json.dumps(doc))
    return dst


def body_equiv(new_path: Path, ref_path: Path, tmp: Path, drift: tuple[str, ...] = ()):
    """perseus.plan.equiv with the contract stamps (and the plan's SUMMARY_DRIFT keys) left out of both."""
    return _equiv.equivalent(str(_without_stamps(new_path, tmp / "new.json", drift)),
                             str(_without_stamps(ref_path, tmp / "ref.json", drift)))


def _out_name(name: str) -> str:
    return f"_regen_{name.replace('/', '_')}"


def replan(name: str, log: Path) -> tuple[Path, float]:
    """Re-plan bootstrap_placements/<name> into bootstrap_placements/_regen_<name> (the caller removes it); returns
    (out dir, wall seconds). Skips (pytest) when the recipe or the graph is not available."""
    plan_dir = BP / name
    spec = GATE.get(name)
    parsed = _recipe(plan_dir)
    graph, recipe, tool = (parsed[0], list(parsed[1]), parsed[2]) if parsed else (None, [], None)
    stages = None
    if spec is not None:
        g2, extra, tool2, stages = spec
        graph = g2 or graph
        recipe += shlex.split(extra)
        tool = tool2 or tool
    elif parsed is None:
        pytest.skip(f"{name} is a measured artifact, not regenerated")
    if graph is None or not ((REPO / graph).is_dir() or Path(graph).is_dir()):
        pytest.skip(f"graph {graph} not present")
    out_name = _out_name(name)
    pypath = str(PKG.parent) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    env = dict(os.environ, GRAPH_DIR=graph, OUT_NAME=out_name, PYTHON=sys.executable,
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", PYTHONPATH=pypath)
    env.update(kv.split("=", 1) for kv in recipe)
    if tool is None:
        cmds = [["bash", str(PKG / "plan" / "run_bootstrap_all_blocks.sh")]]
    elif tool in UPSTREAM:
        pytest.skip(f"{tool}: the released-tool drivers are not ported")
    elif tool in SELF_CONTAINED:
        pytest.skip(f"{tool} captures and plans in one run: run it to regenerate {name}")
    else:
        if stages is None:
            stages = ("blocks",)
            if tool not in LLAMA_TOOLS:
                stages += ("argmax",)
                if (plan_dir / "block_0_feedback_placement.json").exists():
                    stages += ("feedback",)
        cmds = [["bash", str(TOOLS[tool]), graph, out_name] + ([] if s == "blocks" else [s]) for s in stages]
    rc = 0
    t0 = time.perf_counter()
    with log.open("w") as f:
        for cmd in cmds:
            rc = rc or subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0
    assert rc == 0, log.read_text()[-3000:]
    return BP / out_name, wall


def compare(name: str, regen: Path, tmp: Path) -> dict:
    """{block file: {field: differing leaves}}, plus "<file>:body" -> equiv's reason where the body differs and
    "<file>:stamp" where the replanned Config stamp is not the shipped legacy stamp read through the Config."""
    plan_dir = BP / name
    blocks = sorted(plan_dir.glob("block_*_placement.json"))
    assert blocks, f"no block files in {plan_dir}"
    out: dict = {}
    for ref in blocks:
        new = regen / ref.name
        assert new.exists(), f"{name}: no regenerated {ref.name}"
        out[ref.name] = field_diffs(new, ref)
        ok, why = body_equiv(new, ref, tmp, SUMMARY_DRIFT.get(name, ()))
        if not ok:
            out[f"{ref.name}:body"] = why
        # the replanned file carries the Config stamp, equal to the shipped legacy stamp read through the Config
        st_new, st_ref = (json.loads(p.read_text()) for p in (new, ref))
        if "capture_env" in st_new or (st_ref.get("capture_env") is not None
                                       and st_new.get("config") != _contract.stamp_of(st_ref["capture_env"])):
            out[f"{ref.name}:stamp"] = (f"stamp: {st_new.get('config')!r} vs shipped "
                                       f"{st_ref.get('capture_env')!r}")[:600]
    return out


def _assert_identical(name: str, res: dict):
    bad = {k: v for k, v in res.items() if isinstance(v, str) or any(v.values())}
    assert not bad, f"{name}: {json.dumps(bad, indent=1)[:3000]}"


def _run(name: str, tmp_path: Path):
    if name in RETIRED:
        pytest.skip(f"{name}: {RETIRED[name]}")
    try:
        regen, _ = replan(name, tmp_path / "plan.log")
        _assert_identical(name, compare(name, regen, tmp_path))
    finally:
        subprocess.run(["rm", "-rf", str(BP / _out_name(name))])


@pytest.mark.parametrize("name", _plan_dirs())
def test_plan_regenerates(name, tmp_path):
    _run(name, tmp_path)


def _gate_names():
    return [n for n in GATE if n != MAIN] if os.environ.get("PC_PLAN_GATE") == "1" else []


@pytest.mark.parametrize("name", _gate_names())
def test_gate_plan_regenerates(name, tmp_path):
    _run(name, tmp_path)


def main(names: list[str]) -> int:
    """Replan each of `names`, print per-plan field diffs and wall time; exit status = number of failing plans."""
    import tempfile
    failed = 0
    for name in names:
        if name in RETIRED:
            print(f"[gate] {name}: RETIRED ({RETIRED[name]})", flush=True)
            continue
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            logdir = Path(os.environ.get("PLAN_GATE_LOGDIR", td))
            try:
                regen, wall = replan(name, logdir / f"{name}.plan.log")
                res = compare(name, regen, tmp)
            except BaseException as e:  # noqa: BLE001  (pytest.skip is a BaseException)
                print(f"[gate] {name}: ERROR {type(e).__name__}: {str(e)[-1500:]}", flush=True)
                failed += 1
                continue
            finally:
                subprocess.run(["rm", "-rf", str(BP / _out_name(name))])
            files = {k: v for k, v in res.items() if ":" not in k}
            tot = {f: sum(v[f] for v in files.values()) for f in PLAN_FIELDS}
            body = [k for k in res if ":" in k]
            ok = not body and not any(tot.values())
            failed += not ok
            print(f"[gate] {name}: {'IDENTICAL' if ok else 'DIFFERS'} {len(files)} files, {wall:.1f} s, "
                  f"field diffs {tot}, body diffs {len(body)}", flush=True)
            for k in body:
                print(f"[gate]   {k}: {res[k]}", flush=True)
            for k, v in files.items():
                if any(v.values()):
                    print(f"[gate]   {k}: { {f: n for f, n in v.items() if n} }", flush=True)
    return failed


if __name__ == "__main__":
    args = sys.argv[1:] or list(GATE)
    sys.exit(main([n for a in args for n in (TABLE if a == "table" else (a,))]))
