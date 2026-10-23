#!/bin/bash
# Plan this entry's Prefill-128 (CPU only): the 12 blocks and the encrypted tail from one CPU-twin capture
# (run_prefill --fake --tail) with the configs and runtime switches below: gpt2_prefill128_m16s0's runtime plus the GELU
# as one Chebyshev series per block (CONFIGS), the last block on the last position alone (PF_LAST_QUERY), the
# Chebyshev-scaled softmax rounds (SM_CHEB_GS), the packed attention reductions (PF_ATTN_PACK), the last head group's
# K and V in one PCMM output (PF_QKV_PACK) and the SparseB raise drop 1. The blocks take gpt2_prefill128_m16s0's
# recipe on the ILP placer (python/tests/test_paper_plans.py GATE, SparseB landing of the drop); stage 12 (the last
# position into the decode layout, ln_f, the LM head) enters at block 11's planned exit, stage 13 (CutMax) at stage
# 12's, as speedrun/make_tail_plan.sh plans them. The PCMM band shifts are gpt2_prefill128_m16s0's calibration.
#   bash speedrun/make_prefill_plan.sh            -> bootstrap_placements/gpt2_prefill128_pf
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
BASE=gpt2_prefill128_m16s0 OUT=gpt2_prefill128_pf GR=graphs/gpt2_prefill128_pf
CONFIGS=configs/model/approximation/gpt2_base_n32_pf/configs.json
RUNTIME="$(grep "^runtime=" "bootstrap_placements/$BASE/PLAN_CMD.txt" | sed 's/^runtime=//') PF_LAST_QUERY=1 \
FIDESLIB_BTS_RAISE_DROP=512:1 SM_CHEB_GS=1 PF_QKV_PACK=1 PF_ATTN_PACK=3"
RECIPE="PLAN_PRUNE=1 PLAN_SPARSE_BTS_OUT=1:16,512:26 BTS_LEVEL=28 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_LANDING_MAX=42 \
PLAN_RAISE_ROUTES=0 PLAN_REFRESH_ENV_CAP_ROUTES=0:46 PLAN_SLIM_CUBIC=1 PLAN_STC_FIRST_ENTRY=46 \
PLAN_STC_FIRST_FALLBACK=34:1 PLAN_PLACER=ilp PLAN_ILP_TIME_LIMIT=600"
if [ ! -d "$GR/block_13" ]; then
    SETS=$(echo "$RUNTIME" | tr ' ' '\n' | grep = | sed 's/^/--set /')
    PF_PCMM_CALIB_IN="bootstrap_placements/$BASE/pcmm_calib.json" "$PYTHON" -m perseus.examples.gpt2.run_prefill \
        --tokens 128 --fake --tail --capture "$GR" --configs "$CONFIGS" $SETS
fi
rm -rf "bootstrap_placements/$OUT"
TMP="$(mktemp -d graphs/_blocks_XXXX)"
for b in $(seq 0 11); do ln -s "$PWD/$GR/block_$b" "$TMP/block_$b"; done
cp "$GR/capture_env.json" "$TMP/"
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$OUT" blocks
rm -rf "$TMP"
export MAX_LEVEL=50 BTS_LEVEL=28 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20 PLAN_NO_PRESCALE=1 \
    PLAN_ACC_CHAIN=n32 PLAN_MAG_SAFETY=2 PLAN_SPARSE_SLOTS=512,1 PLAN_SPARSE_BTS_OUT=1:16,512:24 PLAN_HINT_ENV_VETO=1 \
    PLAN_PRUNE=1 PLAN_REAL_ROUTE=1 PLAN_RAISE_DROP_SET=1,3,5 PLAN_HINT_VETO_STEPS=tail PLAN_REFRESH_ENV_CAP=50 \
    PLAN_REFRESH_ENV_CAP_STEPS=cutmax:48,tail:46 PLAN_SHARED_REALIZE=1 PLAN_REFRESH_ENV_CAP_ROUTES=0:46 \
    PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_ROUTES=0 PLAN_SLIM_CUBIC=1 PLAN_RAISE_LANDING_MAX=42 PLAN_STC_FIRST_ENTRY=46 \
    PLAN_STC_FIRST_FALLBACK=34:1
read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$OUT/block_11_placement.json")
TMP="$(mktemp -d graphs/_tail_XXXX)"
ln -s "$PWD/$GR/block_12" "$TMP/block_12" && cp "$GR/capture_env.json" "$TMP/"
GRAPH_DIR="$TMP" OUT_NAME="${OUT}_t" FIRST_ENTRY_LEVEL="$L" FIRST_ENTRY_DEG="$D" \
    bash python/perseus/plan/run_bootstrap_all_blocks.sh
cp "bootstrap_placements/${OUT}_t/block_12_placement.json" "bootstrap_placements/$OUT/"
rm -rf "$TMP" "bootstrap_placements/${OUT}_t"
"$PYTHON" - "$GR" "bootstrap_placements/$OUT/block_12_placement.json" <<'PY'
import json, os, sys
from perseus.plan import contract
contract.stamp_file(sys.argv[2], json.load(open(os.path.join(sys.argv[1], "capture_env.json"))))
PY
bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" argmax
cp "bootstrap_placements/$BASE/pcmm_calib.json" "bootstrap_placements/$OUT/"
"$PYTHON" - "$GR" "bootstrap_placements/$OUT/plan_rot_steps.json" <<'PY'
import json, sys
g, f = sys.argv[1:]          # the session's rotation keys: every stage's
steps = set()
for b in range(14):
    for n in json.load(open(f"{g}/block_{b}/graph.json"))["nodes"]:
        steps |= {int(v) for x in n["inputs"] if x.startswith("rot(") for v in x[4:-1].split(",")}
json.dump(sorted(steps), open(f, "w"))
PY
"$PYTHON" - "$GR" <<'PY'
import glob, json, sys
for f in glob.glob(f"{sys.argv[1]}/**/capture_config.json", recursive=True):    # the capturing machine's paths out
    c = json.load(open(f))
    for sec in (v for v in c.values() if isinstance(v, dict)):
        for k in [k for k, v in sec.items() if isinstance(v, str) and (v.startswith("/") or k == "pythonpath")]:
            del sec[k]
    open(f, "w").write(json.dumps(c, indent=2))
PY
PLANNER="$(git log -1 --format=%h -- python/perseus/plan) (python/perseus/plan's last commit; PLAN_PLACER=ilp)"
printf '%s\n' "graph=$GR" \
    "tool=speedrun/make_prefill_plan.sh (make_plan.sh blocks + argmax, the tail as make_tail_plan.sh)" \
    "planner=$PLANNER" "recipe=$RECIPE" "runtime=$RUNTIME" "configs=$CONFIGS" \
    "pcmm_calib=$BASE's (the PCMM band shifts)" > "bootstrap_placements/$OUT/PLAN_CMD.txt"
ls "bootstrap_placements/$OUT"
