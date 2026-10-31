#!/bin/bash
# Prefill-128's PCMM finish raise drops (CPU only): the served plan's program with the finishes of the qkv and up
# linears raised below the chain top where their consumers have the slack (the planner's deliberate-drop probe,
# PLAN_RAISE_DELIB=1 on the served rd pass: qkv output 2 by 5 levels in every block, outputs 1 and 4 by 4-5 in blocks
# 1, 5-10, every output in blocks 3 (1) and 11 (5); every up output by 1, by 3 in blocks 2 and 11). The switches
# PF_PCMM_RAISE / PF_PCMM_RAISE_OUT below set those drops on the finishes' descriptors; the capture's PCMM finishes are
# landed where they land under them (a copy of GR, fakecap's landing), and the served rd pass
# (PLAN_KEEP_PLACEMENTS_EXACT of ${BASE}'s program, its raise-drop search) re-plans blocks 0-11 on it: the same
# placements, hints, routes, factors, refresh drops and block exits, every level downstream of a dropped finish
# deeper. Stages 12-13 are the served plan's.
#   bash speedrun/make_prefill_pcmm_drops.sh   -> bootstrap_placements/gpt2_prefill128_n32a4_pfbhrd_cm (in place; the
#   served runtime line also carries FIDESLIB_CTS_ALT_BUDGET=5: the dense raise variants of drop >= 1 on a 5-stage CtS)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
BASE=gpt2_prefill128_n32a4_pfbh PLAN=gpt2_prefill128_n32a4_pfbhrd_cm GR=graphs/gpt2_prefill128_pfbh SUF=_pcmmrd
export PF_PCMM_RAISE="up:3:2-2,up:3:11-11,up:1,qkv:5:11-11,qkv:1:3-3"
export PF_PCMM_RAISE_OUT="qkv:2:5,qkv:1:4:1-1,qkv:4:4:1-1,qkv:1:4:5-5,qkv:4:4:5-5,qkv:1:5:6-7,qkv:4:5:6-7,qkv:1:4:8-8,\
qkv:4:4:8-8,qkv:1:5:9-9,qkv:4:5:9-9,qkv:1:4:10-10,qkv:4:4:10-10"
rm -rf "$GR$SUF" && cp -r "$GR" "$GR$SUF"
"$PYTHON" - "$GR$SUF" <<'PY'
import json, sys, types
from perseus.impl import pcmm
d = sys.argv[1]
n_ed = 0
for b in range(12):
    f = f"{d}/block_{b}/graph.json"
    g = json.load(open(f))
    for n in g["nodes"]:
        ext = (n.get("args") or {}).get("external") or {}
        if n["op_type"] != "deliberate_bootstrap" or ext.get("op") != "pcmm":
            continue
        k = pcmm._drop_of(types.SimpleNamespace(name=ext["linear"], n_out=ext["n_out"]), ext["out"])
        if k:
            ext["raise_drop"] = k
            n["output_level"] += 2 * k
            n["args"]["landing"][0] += 2 * k
            n["shadow"]["level"] += 2 * k
            n_ed += 1
    json.dump(g, open(f, "w"))
print(f"{d}: {n_ed} PCMM finishes dropped")
PY
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")"
OUT=_pcmmrd
rm -rf "bootstrap_placements/$OUT"
TMP="$(mktemp -d graphs/_blocks_XXXX)"
for b in $(seq 0 11); do ln -s "$PWD/$GR$SUF/block_$b" "$TMP/block_$b"; done
cp "$GR/capture_env.json" "$TMP/"
# shellcheck disable=SC2086
env $RECIPE PLAN_KEEP_PLACEMENTS=bootstrap_placements/$BASE bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$OUT" blocks
rm -rf "$TMP"
"$PYTHON" - "bootstrap_placements/$PLAN" "bootstrap_placements/$OUT" <<'PY'
import json, sys
a, b = sys.argv[1:]
keys = ("placements", "hint_fire", "sparse_slots", "correction_factor", "raise_drop", "real_route", "rescale_after")
for k in range(12):
    x, y = (json.load(open(f"{d}/block_{k}_placement.json")) for d in (a, b))
    bad = [f for f in keys if x.get(f) != y.get(f)]
    ex = lambda s: (s["exit_level"], s.get("exit_deg") or 1)
    if bad or ex(x["summary"]) != ex(y["summary"]):
        sys.exit(f"block {k}: the finishes' drops changed {bad or 'the exit'}")
PY
for k in $(seq 0 11); do cp "bootstrap_placements/$OUT/block_${k}_placement.json" "bootstrap_placements/$PLAN/"; done
rm -rf "bootstrap_placements/$OUT"
rm -rf "$GR$SUF"
sed -i -e "/^runtime=/{/PF_PCMM_RAISE=/!s|\$| PF_PCMM_RAISE=$PF_PCMM_RAISE PF_PCMM_RAISE_OUT=$PF_PCMM_RAISE_OUT|}" \
    -e "/^pcmm_drops=/d" "bootstrap_placements/$PLAN/PLAN_CMD.txt"
echo "pcmm_drops=blocks 0-11 by speedrun/make_prefill_pcmm_drops.sh: $BASE's program re-planned (the rd pass) on $GR \
with the qkv and up finishes landed where PF_PCMM_RAISE / PF_PCMM_RAISE_OUT drop them" >> "bootstrap_placements/$PLAN/PLAN_CMD.txt"
