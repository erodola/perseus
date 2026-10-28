#!/bin/bash
# CutMax with the scaled normalisers (stage 13) on top of a plan's other stages (CPU only): OUT gets BASE's stages
# 0-12 byte for byte and stage 13 planned by make_plan.sh argmax at BASE's stage-12 exit, from the committed capture
# (its stage 13 carries the CutMax magnitudes over 211 logit rows, capture_mags --rows): Decode-8 with BASE's recipe on
# the min-cut placer, Prefill-128 (TRACK=prefill) with speedrun/make_prefill_plan.sh's argmax env. CONFIGS (default
# gpt2_base_n32_cm / gpt2_base_n32_pf_cm) must be BASE's configs with that cutmax section, and the capture binds when
# BASE's runtime shapes the op stream as the capture's did (stage 13's stamp equals stage 12's); both are checked.
# CUTMAX_GRAPH: another Decode-8 capture whose stage 13 has those magnitudes, taken under BASE's runtime.
# CUTMAX_PLACER=ilp: Decode-8's stage 13 on the recipe's ILP placer instead of the min-cut.
#   bash speedrun/make_cutmax_plan.sh gpt2_decode_python_n32_m16i_fcf gpt2_decode_python_n32_m16i_fcf_cm
#   TRACK=prefill bash speedrun/make_cutmax_plan.sh gpt2_prefill128_pf gpt2_prefill128_pf_cm
#   TRACK=prefill GR=graphs/gpt2_prefill128_pfbh_cm bash speedrun/make_cutmax_plan.sh gpt2_prefill128_pfbh \
#       gpt2_prefill128_pfbh_cm      (a capture of BASE's runtime: speedrun/make_prefill_plan.sh's cm step)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
BASE=${1:?base plan} OUT=${2:?out name}
if [ "${TRACK:-decode}" = prefill ]; then
    CM=configs/model/approximation/gpt2_base_n32_pf_cm/configs.json GR=${GR:-graphs/gpt2_prefill128_pf_cm}
    # make_prefill_plan.sh's argmax env, at BASE's chain's deepest user level (n32's 50; 48 on the lean n32a4, which
    # BASE's runtime line names)
    TOP=50
    if grep -q "^runtime=.*CHAIN_PRESET=n32a4" "bootstrap_placements/$BASE/PLAN_CMD.txt"; then TOP=48; fi
    export MAX_LEVEL=$TOP BTS_LEVEL=28 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20 \
        PLAN_NO_PRESCALE=1 PLAN_ACC_CHAIN=n32 PLAN_MAG_SAFETY=2 PLAN_SPARSE_SLOTS=512,1 \
        PLAN_SPARSE_BTS_OUT=1:16,512:24 PLAN_HINT_ENV_VETO=1 PLAN_PRUNE=1 PLAN_REAL_ROUTE=1 \
        PLAN_RAISE_DROP_SET=1,3,5 PLAN_HINT_VETO_STEPS=tail \
        PLAN_REFRESH_ENV_CAP=$TOP PLAN_REFRESH_ENV_CAP_STEPS=cutmax:48,tail:46 PLAN_SHARED_REALIZE=1 \
        PLAN_REFRESH_ENV_CAP_ROUTES=0:46 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_ROUTES=0 PLAN_SLIM_CUBIC=1 \
        PLAN_RAISE_LANDING_MAX=42 PLAN_STC_FIRST_ENTRY=46 PLAN_STC_FIRST_FALLBACK=34:1
    RECIPE="" HOW="make_prefill_plan.sh's argmax env"
else
    CM=configs/model/approximation/gpt2_base_n32_cm/configs.json GR=${CUTMAX_GRAPH:-graphs/gpt2_decode_python_n32_cm}
    RECIPE=$(sed -n 's/^recipe=//p' "bootstrap_placements/$BASE/PLAN_CMD.txt")
    if [ "${CUTMAX_PLACER:-min_cut}" = ilp ]; then
        # the recipe's own placer (ILP, its quality weight and tie-break) on stage 13 as on the blocks
        RECIPE=$(echo "$RECIPE" | sed 's/PLAN_ARGMAX_MINCUT=[^ ]*//')
        HOW="the recipe above on the ILP placer (CUTMAX_PLACER=ilp)"
    else
        RECIPE=$(echo "$RECIPE" | sed 's/PLAN_PLACER=[^ ]*//; s/PLAN_ILP_TIME_LIMIT=[^ ]*//')
        HOW="the recipe above on the min-cut placer"
    fi
fi
CONFIGS=${3:-$CM}
"$PYTHON" - "bootstrap_placements/$BASE" "$CONFIGS" "$CM" <<'PY'
import json, sys
from perseus._paths import PKG
from perseus.examples.gpt2 import env
base, configs, cm = sys.argv[1:]
load = lambda p: json.load(open(PKG / p))
b, c = load(env.plan_configs(base, "n32")), load(configs)
if {k: v for k, v in b.items() if k != "cutmax"} != {k: v for k, v in c.items() if k != "cutmax"}:
    sys.exit(f"{configs}: not {base}'s configs outside the cutmax section (stages 0-12 would not bind)")
if c["cutmax"] != load(cm)["cutmax"]:
    sys.exit(f"{configs}: the cutmax section is not {cm}'s (the capture of stage 13 would not bind)")
PY
rm -rf "bootstrap_placements/$OUT" && cp -r "bootstrap_placements/$BASE" "bootstrap_placements/$OUT"
rm -f "bootstrap_placements/$OUT/block_13_placement.json"
# shellcheck disable=SC2086
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" argmax
"$PYTHON" - "bootstrap_placements/$OUT" "$CONFIGS" "$GR" "$HOW" <<'PY'
import json, sys
d, configs, gr, how = sys.argv[1:]
rows = "graphs/gpt2_decode_python_n32_cm/cutmax_rows.json"
a, b = (json.load(open(f"{d}/block_{k}_placement.json")).get("config") for k in (12, 13))
if a != b:
    sys.exit(f"{d}: stage 13's capture stamp differs from stage 12's: recapture under this plan's runtime and fill its "
             f"stage 13 with capture_mags --argmax --rows (the logit rows of {rows}, scripts/utils/cutmax_rows.py)")
lines = [l for l in open(f"{d}/PLAN_CMD.txt").read().splitlines() if not l.startswith("cutmax=")]
lines = [f"configs={configs}{l[len(l.split()[0]):]}" if l.startswith("configs=") else l for l in lines]
lines = [f"{l} (stages 0-12; stage 13: {gr})" if l.startswith("graph=") else l for l in lines]
p = json.load(open(f"{d}/block_13_placement.json"))
s = p["summary"]
lines.append(f"cutmax=the lines above are the base plan's and describe its stages 0-12 (a stage-13 site they name "
             f"is the base's argmax, replaced here); stage 13 is CutMax with the scaled normalisers, replanned by "
             f"speedrun/make_cutmax_plan.sh from {gr} ({how}, at stage 12's exit): {s['total_bootstraps']} refreshes, "
             f"CF {p.get('correction_factor')}; its magnitudes are the per-node max over the logit rows of the windows "
             f"in {rows} (OpenWebText and C4 validation shard 00001, never the dev prompts; "
             f"scripts/utils/cutmax_rows.py, capture_mags --argmax --rows)")
open(f"{d}/PLAN_CMD.txt", "w").write("\n".join(lines) + "\n")
print(f"{d}: stage 13 {s['total_bootstraps']} refreshes, exit {s['exit_level']}:{s.get('exit_deg')}")
PY
