#!/bin/bash
# Plan Prefill-128's encrypted tail on top of the shipped GPT-2 prefill plan (CPU only): stage 12 moves the last
# position into the decode layout and runs ln_f and the LM head, stage 13 is CutMax. Their capture comes from the
# CPU twin with the plan's runtime (run_prefill --fake --tail); stage 12 enters at block 11's planned exit, stage 13
# at stage 12's (make_plan.sh's argmax pass). The recipe is the decode tail's, with the prefill plan's landings.
#   bash speedrun/make_tail_plan.sh            -> bootstrap_placements/gpt2_prefill128_m16s0_tail
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
BASE=gpt2_prefill128_m16s0 OUT=gpt2_prefill128_m16s0_tail GR=graphs/gpt2_prefill128_m16s0_tail
if [ ! -d "$GR/block_13" ]; then
    SETS=$(grep "^runtime=" "bootstrap_placements/$BASE/PLAN_CMD.txt" | sed 's/^runtime=//' | tr ' ' '\n' | grep = | sed 's/^/--set /')
    PF_PCMM_CALIB_IN="bootstrap_placements/$BASE/pcmm_calib.json" "$PYTHON" -m perseus.examples.gpt2.run_prefill \
        --tokens 128 --fake --tail --capture "$GR" $SETS
fi
export MAX_LEVEL=50 BTS_LEVEL=28 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20 PLAN_NO_PRESCALE=1 \
    PLAN_ACC_CHAIN=n32 PLAN_MAG_SAFETY=2 PLAN_SPARSE_SLOTS=512,1 PLAN_SPARSE_BTS_OUT=1:16,512:24 PLAN_HINT_ENV_VETO=1 \
    PLAN_PRUNE=1 PLAN_REAL_ROUTE=1 PLAN_RAISE_DROP_SET=1,3,5 PLAN_HINT_VETO_STEPS=tail PLAN_REFRESH_ENV_CAP=50 \
    PLAN_REFRESH_ENV_CAP_STEPS=cutmax:48,tail:46 PLAN_SHARED_REALIZE=1 PLAN_REFRESH_ENV_CAP_ROUTES=0:46 \
    PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_ROUTES=0 PLAN_SLIM_CUBIC=1 PLAN_RAISE_LANDING_MAX=42 PLAN_STC_FIRST_ENTRY=46 \
    PLAN_STC_FIRST_FALLBACK=34:1
rm -rf "bootstrap_placements/$OUT" && cp -r "bootstrap_placements/$BASE" "bootstrap_placements/$OUT"
read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$BASE/block_11_placement.json")
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
printf 'tail=stages 12 (token_of + ln_f + LM head) and 13 (CutMax) from %s, planned by speedrun/make_tail_plan.sh\n' \
    "$GR" >> "bootstrap_placements/$OUT/PLAN_CMD.txt"
ls "bootstrap_placements/$OUT"
