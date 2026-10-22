"""Derive the pc-config presets from the old env dicts: the exact environment each run path of the port exports.

    PYTHONPATH=python .venv/bin/python crates/pc-config/tools/derive_presets.py > crates/pc-config/src/presets_data.rs

Each preset replays, in a clean environment, what the perseus C++ stack did before a session opened: `source
scripts/local_env.sh` for CHAIN=n32 (defaults; only the names a Config field owns are kept), then the example's own
steps (`env.export_env` forces `env.ENV` + the plan's `runtime=` line, the fake capture's `fakecap.session_env`
applies `SessionProfile.gpt2_decode_n32` as defaults; the prefill paths add their PF_* / StC-first overrides).
The FAKECAP_LANDINGS value is the one the shipped capture of that path was made with.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_fields import RENAME, rows  # noqa: E402

# scripts/local_env.sh, CHAIN=n32 branch (perseus repo), the names it exports by default
LOCAL_ENV_N32 = {
    "CHAIN": "n32", "OMP_NUM_THREADS": "16", "MALLOC_ARENA_MAX": "2", "LOGN": "16", "BTS_ITERATIONS": "1",
    "COMPOSITE_DEGREE": "2", "CORRECTION_FACTOR": "6", "AUTO_BTS_LEVEL": "46", "FIDESLIB_ROT_KEY_BAND": "22",
    "FHE_STAGE_ARENA_GB": "24", "KV_ARENA_GB": "24", "FHE_LMHEAD_CAP": "44", "CACHE_READ_LEVEL_K": "34",
    "CACHE_READ_LEVEL_V": "34", "SPARSE_BTS_SLOTS": "512,1", "SPARSE_AUTO": "2", "FUSED_SM_DEN": "1",
    "FUSED_LN_VAR": "0", "PLAN_SPARSE_SLOTS": "512,1", "FHE_PT_COEFF_ENCODE": "0",
}

PLANS = "bootstrap_placements"

# The runtime= lines of the two 64-token PCMM StC-first prefill plans the prefill presets were derived from (no longer
# shipped): the presets keep them, so the result plans layered on top run as before.
RETIRED = {
    "64-token GPT-2 PCMM prefill": (
        "FIDESLIB_BTS_SHIFT=1 FIDESLIB_SPRU=64 FIDESLIB_BTS_RAISE_DROPS=1,3,5 SM_DEN_RECIP=1 FIDESLIB_BTS_REAL=1 "
        "GELU_FOLD=1 LN_CHEB=1 SM_GS_FIRST=1 SM_PERIODIC=1 SM_FOLD=1 FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32 "
        "KV_LANES=1 CHEB_BSGS=1 QKV_LANES=1  SM_HRS_PACK=1 CUTMAX_PAIR=1 IMPL_DEAD_OPS=1 LADDER_RADIX_BITS=3 "
        "FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 AUTO_BTS_LEVEL=44 "
        "BTS_MAX_INPUT_LEVEL=48 PF_LINEAR=pcmm PF_ACT_FOLD=1 PF_GELU_JOINT=1 PF_JOINT_FORCE=1 PF_SM_JOINT=1 "
        "PF_PCMM_STAGED_ONLY=0 PF_PCMM_CF=6"),
    "Llama PCMM prefill": (
        "FIDESLIB_BTS_SHIFT=1 FIDESLIB_SPRU=64 FIDESLIB_BTS_RAISE_DROPS=1,3,5 SM_DEN_RECIP=1 FIDESLIB_BTS_REAL=1 "
        "GELU_FOLD=1 LN_CHEB=1 SM_GS_FIRST=1 SM_PERIODIC=2 SM_FOLD=1 FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32 "
        "KV_LANES=1 CHEB_BSGS=1 QKV_LANES=1 BTS_MAX_INPUT_LEVEL=50 IMPL_SCOPED_NAMES=1 SPARSE_BTS_SLOTS=256,1 "
        "FIDESLIB_BTS_SPARSE_B_MAX=1024 SPARSE_LEVEL_BUDGET=3,3 SM_HRS_PACK=1 CUTMAX_PAIR=1 IMPL_DEAD_OPS=1 "
        "LADDER_RADIX_BITS=3 FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 "
        "PF_LINEAR=pcmm PF_ACT_FOLD=1 PF_GELU_JOINT=1 PF_JOINT_FORCE=1 PF_PCMM_WBITS=24 PF_PCMM_STAGED_ONLY=0 "
        "PF_PCMM_CF=6"),
}

# (preset, plan the runtime line comes from (a plan dir, or a RETIRED key), flow, FAKECAP_LANDINGS of the shipped
# capture)
PRESETS = [
    ("n32_gpt2_decode", "gpt2_decode", "gpt2_decode", "0:32,1:14,256:22,512:24"),
    ("n32_gpt2_prefill", "64-token GPT-2 PCMM prefill", "gpt2_prefill", "0:32,1:14,256:22,512:22"),
    ("n32_llama_decode", "llama31_8b_decode", "llama_decode", None),
    ("n32_llama_prefill", "Llama PCMM prefill", "llama_prefill", None),
]

FLOW = r'''
import json, os, sys
plan, flow = sys.argv[1], sys.argv[2]
if flow.startswith("gpt2"):
    from perseus.examples.gpt2 import env
else:
    from perseus.examples.llama import env
from perseus.impl import fakecap
if flow == "gpt2_decode":
    over = {**env.plan_runtime(plan, "n32"), "GPT2_PACKING": "cachemir_complex"}
    env.export_env(3, chain="n32", CKKS_COMPLEX="1", **over)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over}, "n32", True)
elif flow in ("gpt2_prefill", "llama_prefill"):
    os.environ.setdefault("KV_ARENA_GB", "0")
    pack = {"GPT2_PACKING": "cachemir"} if flow == "gpt2_prefill" else {"LLAMA_PACKING": "cachemir_complex"}
    over = {"FIDESLIB_ENCODE_NOSYNC": "0", **env.plan_runtime(plan, "n32"), **pack}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    if os.environ.get("PF_LINEAR", "") == "pcmm" or os.environ.get("PF_ATTN", "") == "ccmm":
        over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
                "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(3, chain="n32", CKKS_COMPLEX="1", **over)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
elif flow == "llama_decode":
    over = {**env.plan_runtime(plan, "n32"), "LLAMA_PACKING": "cachemir_complex"}
    env.export_env(3, chain="n32", **{"CKKS_COMPLEX": "1", **over})
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
json.dump(dict(os.environ), sys.stdout)
'''


def main():
    names = {RENAME.get(r["name"], r["field"]).split(" ")[0]: r["name"] for r in rows()}
    known = {env for field, env in names.items() if not field.startswith("(")}
    out = []
    tmp = tempfile.TemporaryDirectory()
    for preset, plan, flow, landings in PRESETS:
        env0 = {"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT / "python"), "PYTHONDONTWRITEBYTECODE": "1",
                **LOCAL_ENV_N32}
        plan_dir = f"{PLANS}/{plan}"
        if plan in RETIRED:         # a plan dir holding just its runtime= line (all the flows read of a plan)
            plan_dir = Path(tmp.name) / preset
            plan_dir.mkdir()
            (plan_dir / "PLAN_CMD.txt").write_text(f"runtime={RETIRED[plan]}\n")
        res = subprocess.run([sys.executable, "-c", FLOW, str(plan_dir), flow], env=env0, cwd=ROOT,
                             capture_output=True, text=True, check=True)
        env = json.loads(res.stdout)
        for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH"):     # the GPU guard stays env; PYTHONPATH is this tool's
            env.pop(k, None)
        if landings:
            env["FAKECAP_LANDINGS"] = landings
        kv = sorted((k, v) for k, v in env.items() if k in known)
        out.append((preset, plan, flow, kv))
    w = sys.stdout.write
    w("// @generated by crates/pc-config/tools/derive_presets.py -- the env each perseus run path exported\n\n")
    for preset, plan, flow, kv in out:
        src = f"the {plan} plan" if plan in RETIRED else f"`{plan}`"
        w(f"/// `{preset}`: local_env.sh (n32) + the `{flow}` path of the port, runtime of {src}.\n")
        w(f"pub const {preset.upper()}: &[(&str, &str)] = &[\n")
        for k, v in kv:
            w(f"    ({json.dumps(k)}, {json.dumps(v)}),\n")
        w("];\n\n")


if __name__ == "__main__":
    main()
