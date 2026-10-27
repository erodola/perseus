#!/bin/bash
# Plan this entry's Prefill-128 (CPU only): the 12 blocks and the encrypted tail from a CPU-twin capture
# (run_prefill --fake --tail) with the configs and runtime switches below: gpt2_prefill128_m16s0's runtime plus the GELU
# as one Chebyshev series per block (CONFIGS), the last block on the last position alone (PF_LAST_QUERY), the
# Chebyshev-scaled softmax rounds (SM_CHEB_GS), the packed attention reductions (PF_ATTN_PACK) with baby-step
# giant-step lane rotations (PF_ATTN_BSGS), one softmax for both query chunks of the half-empty head group
# (PF_SM_HPACK), the last head group's K and V in one PCMM output (PF_QKV_PACK) and the SparseB raise drop 1. The
# blocks take gpt2_prefill128_m16s0's recipe on the ILP placer (python/tests/test_paper_plans.py GATE, SparseB landing
# of the drop) with an ILP time limit no block's solve comes near and the tie-break; stage 12 (the last position into
# the decode layout, ln_f, the LM head) enters at block 11's planned exit, stage 13 (CutMax) at stage 12's, as
# speedrun/make_tail_plan.sh plans them (speedrun/make_cutmax_plan.sh then replans stage 13 with CutMax's scaled
# normalisers: the _cm plan). The capture and the PCMM band shifts come from four rows of the calibration pool. The
# served plan, ${OUT}rd_cm, keeps every block's program (placements, fired hints, realize anchors) and raises refreshes
# below the chain top where their landings have slack (the planner's raise-drop search over the dense, SparseB-512
# and SPRU routes, drops 1-5 on SparseB's base drop 1).
#   bash speedrun/make_prefill_plan.sh  -> bootstrap_placements/gpt2_prefill128_pfbh{,_cm}, ..._pfbhrd{,_cm}
#   RESUME=1 bash speedrun/make_prefill_plan.sh   continues a run that stopped (a preempted job): the rows' oracles, the
#   merged capture, the blocks already planned (PLAN_RESUME) and the CutMax capture are kept, each written whole, and
#   so are the later stages it completed (stage 12, the argmax, each CutMax replan, the raise-drop pass's blocks)
#   PRESET=n32a4 bash speedrun/make_prefill_plan.sh   the plan for the lean chain (pc-core ChainSpec::n32a4, which the
#   server runs on the plan's CHAIN_PRESET): the same captures, every value and refresh at level 48 or below (n32a4's
#   deepest user level, the composite pair of its 2nd and 3rd bottom primes; n32's is 50), and the inputs of every
#   PCMM of band shift above 1 one level above the StC-first entry (PLAN_PCMM_PREDIV_CAP: the runtime divides the
#   shift on that level, pc-gpu matmul::pcmm::prediv), the softmax steps sized with SM_KAPPA 6
#   -> bootstrap_placements/gpt2_prefill128_n32a4_pfbh{,_cm}, ..._pfbhrd{,_cm}
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
case ${PRESET:=n32} in
    n32) TOP=50 TAG= CAP= ;;
    n32a4) TOP=48 TAG=n32a4_ CAP=" MAX_LEVEL=48 PLAN_REFRESH_ENV_CAP=48 PLAN_PCMM_PREDIV_CAP=42"
        # CutMax's stage 13 on the ILP (its added refresh where the quality weight puts it, as Decode-8's)
        export PLAN_PCMM_PREDIV_CAP=42 CUTMAX_PLACER=ilp
        # its plan refreshes squares of the softmax's normalising rounds that n32's does not (under 4x, block 1's v_1402
        # at CF 2: 1.17x of its band over the range gate's 80 C4 prompts)
        : "${SM_KAPPA:=6}" ;;
    *) echo "PRESET=$PRESET: n32 or n32a4" >&2; exit 2 ;;
esac
BASE=gpt2_prefill128_m16s0 OUT=gpt2_prefill128_${TAG}pfbh GR=graphs/gpt2_prefill128_pfbh
CONFIGS=configs/model/approximation/gpt2_base_n32_pf/configs.json
RUNTIME="$(grep "^runtime=" "bootstrap_placements/$BASE/PLAN_CMD.txt" | sed 's/^runtime=//') PF_LAST_QUERY=1 \
FIDESLIB_BTS_RAISE_DROP=512:1 SM_CHEB_GS=1 PF_QKV_PACK=1 PF_ATTN_PACK=3 PF_ATTN_BSGS=1 PF_SM_HPACK=1"
# the switch the server alone reads (pc-gpu; the CPU capture and the planner never see it): EvalMod's relinearization
# and the rescale after it as one exact conversion
SERVER_RUNTIME="PC_EM_FUSE_RESCALE=1${TAG:+ CHAIN_PRESET=$PRESET}"
# replay switches, set on the plan's runtime= line only (neither the capture nor the placements read them): refresh pairs,
# the SparseB routes' EvalMod series
REPLAY="PC_BTS_PAIR=1 FIDESLIB_BTS_SPARSE_B_K=16"
# The sites that carry the last position alone (block 11's attention on the last query, the tail) see one value per
# capture row, not a maximum over 128 positions: they are sized with 4x the drift margin (PLAN_MAG_SAFETY_STEPS). The
# softmax's normalising rounds square their input again and again, so a drift r at the head of the chain is r^2, r^4,
# r^8 at its squares: the softmax steps of blocks 0-10 are sized with SM_KAPPA (2x the default)
SM_KAPPA=${SM_KAPPA:-4}
MAGS="PLAN_MAG_SAFETY_STEPS=blk11.attn:8,tail:8$(for b in $(seq 0 10); do printf ',blk%d.attn.softmax:%s' "$b" "$SM_KAPPA"; done)"
RECIPE="PLAN_PRUNE=1 PLAN_SPARSE_BTS_OUT=1:16,512:26 BTS_LEVEL=28 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_LANDING_MAX=42 \
PLAN_RAISE_ROUTES=0 PLAN_REFRESH_ENV_CAP_ROUTES=0:46 PLAN_SLIM_CUBIC=1 PLAN_STC_FIRST_ENTRY=46 \
PLAN_STC_FIRST_FALLBACK=34:1 PLAN_PLACER=ilp PLAN_ILP_TIME_LIMIT=3600 PLAN_ILP_TIEBREAK=1 $MAGS$CAP"
# The capture and the PCMM band shifts come from ROWS rows of the OpenWebText calibration pool (the rows of
# speedrun/make_decode_capture.sh; each row's oracle by scripts/utils/gen_gpt2_oracle.py): a CPU-twin prefill of each
# row records every linear's max |y| (PF_PCMM_CALIB; the shifts, impl/pcmm.py band_shift, from the largest over the
# rows: $GR/pcmm_calib.json), a second one with those shifts captures it, and the plan's capture is the rows' captures
# max-merged by name (scripts/utils/merge_mags.py): a refresh is sized for the largest value any row gives it.
# POOL: the pool .npy, by default the one perseus.calibrate.data builds (<HF_HOME>/perseus/pools/
# openwebtext_gpt2_2000000.npy), as speedrun/make_decode_capture.sh; its rows at the ROWS offsets are the owt_pool
# windows of graphs/gpt2_decode_python_n32_cm/cutmax_rows.json (the first 16 tokens each), against which a pool can
# be checked.
POOL=${POOL:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/perseus/pools/openwebtext_gpt2_2000000.npy}
ROWS=${ROWS:-4096 8192 12288 16384}
SETS=$(echo "$RUNTIME" | tr ' ' '\n' | grep = | sed 's/^/--set /')
if [ ! -d "$GR/block_13" ]; then
    for r in $ROWS; do
        d="$PERSEUS_DATA/oracle/gpt2/owt_$r"
        [ -d "$d" ] || { rm -rf "$d.part" && "$PYTHON" scripts/utils/gen_gpt2_oracle.py \
            --model openai-community/gpt2 --pool "$POOL" --pool-offset "$r" --T 128 --out "$d.part" && mv "$d.part" "$d"; }
    done
    TMP="$(mktemp -d graphs/_rows_XXXX)"
    pids=()
    for r in $ROWS; do
        ALL_BLOCKS_IO_DIR="$PERSEUS_DATA/oracle/gpt2/owt_$r" PF_PCMM_CALIB="$TMP/calib_$r.json" "$PYTHON" \
            -m perseus.examples.gpt2.run_prefill --tokens 128 --fake --tail --configs "$CONFIGS" $SETS \
            > "$TMP/calib_$r.log" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "a row's calibration failed: $TMP/calib_*.log" >&2; exit 1; }; done
    "$PYTHON" - "$TMP/pcmm_calib.json" "$TMP"/calib_*.json <<'PY'
import json, sys
out = {}
for f in sys.argv[2:]:
    for k, v in json.load(open(f)).items():
        out[k] = max(out.get(k, 0.0), v)
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
    pids=()
    for r in $ROWS; do
        ALL_BLOCKS_IO_DIR="$PERSEUS_DATA/oracle/gpt2/owt_$r" PF_PCMM_CALIB_IN="$TMP/pcmm_calib.json" "$PYTHON" \
            -m perseus.examples.gpt2.run_prefill --tokens 128 --fake --tail --capture "$TMP/r$r" --configs "$CONFIGS" \
            $SETS > "$TMP/r$r.log" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "a row's capture failed: $TMP/r*.log" >&2; exit 1; }; done
    first=${ROWS%% *}
    rm -rf "$GR.part"
    "$PYTHON" scripts/utils/merge_mags.py "$TMP/r$first" "$GR.part" $(for r in $ROWS; do echo "$TMP/r$r"; done)
    cp "$TMP/pcmm_calib.json" "$GR.part/" && rm -rf "$GR" && mv "$GR.part" "$GR" && rm -rf "$TMP"
fi
RD=${OUT}rd
# the later stages a run completed (RESUME), marked in ST once written whole
ST=graphs/_resume_$OUT
[ "${RESUME:-0}" = 1 ] || rm -rf "bootstrap_placements/$OUT" "bootstrap_placements/$RD" "$ST"
mkdir -p "$ST"
TMP="$(mktemp -d graphs/_blocks_XXXX)"
for b in $(seq 0 11); do ln -s "$PWD/$GR/block_$b" "$TMP/block_$b"; done
cp "$GR/capture_env.json" "$TMP/"
env $RECIPE PLAN_RESUME="${RESUME:-0}" bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$OUT" blocks
rm -rf "$TMP"
export MAX_LEVEL=$TOP BTS_LEVEL=28 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20 PLAN_NO_PRESCALE=1 \
    PLAN_ACC_CHAIN=n32 PLAN_MAG_SAFETY=2 PLAN_SPARSE_SLOTS=512,1 PLAN_SPARSE_BTS_OUT=1:16,512:24 PLAN_HINT_ENV_VETO=1 \
    PLAN_PRUNE=1 PLAN_REAL_ROUTE=1 PLAN_RAISE_DROP_SET=1,3,5 PLAN_HINT_VETO_STEPS=tail PLAN_REFRESH_ENV_CAP=$TOP \
    PLAN_REFRESH_ENV_CAP_STEPS=cutmax:48,tail:46 PLAN_SHARED_REALIZE=1 PLAN_REFRESH_ENV_CAP_ROUTES=0:46 \
    PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_ROUTES=0 PLAN_SLIM_CUBIC=1 PLAN_RAISE_LANDING_MAX=42 PLAN_STC_FIRST_ENTRY=46 \
    PLAN_STC_FIRST_FALLBACK=34:1 "$MAGS"
if [ ! -e "$ST/tail" ]; then
    read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$OUT/block_11_placement.json")
    TMP="$(mktemp -d graphs/_tail_XXXX)"
    ln -s "$PWD/$GR/block_12" "$TMP/block_12" && cp "$GR/capture_env.json" "$TMP/"
    rm -rf "bootstrap_placements/${OUT}_t"
    GRAPH_DIR="$TMP" OUT_NAME="${OUT}_t" FIRST_ENTRY_LEVEL="$L" FIRST_ENTRY_DEG="$D" \
        bash python/perseus/plan/run_bootstrap_all_blocks.sh
    cp "bootstrap_placements/${OUT}_t/block_12_placement.json" "bootstrap_placements/$OUT/"
    rm -rf "$TMP" "bootstrap_placements/${OUT}_t"
    "$PYTHON" - "$GR" "bootstrap_placements/$OUT/block_12_placement.json" <<'PY'
import json, os, sys
from perseus.plan import contract
contract.stamp_file(sys.argv[2], json.load(open(os.path.join(sys.argv[1], "capture_env.json"))))
PY
    touch "$ST/tail"
fi
if [ ! -e "$ST/argmax" ]; then
    bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" argmax
    touch "$ST/argmax"
fi
cp "$GR/pcmm_calib.json" "bootstrap_placements/$OUT/"
"$PYTHON" - "$GR" "bootstrap_placements/$OUT/plan_rot_steps.json" <<'PY'
import json, sys
g, f = sys.argv[1:]          # the session's rotation keys: every stage's
steps = set()
for b in range(14):
    for n in json.load(open(f"{g}/block_{b}/graph.json"))["nodes"]:
        steps |= {int(v) for x in n["inputs"] if x.startswith("rot(") for v in x[4:-1].split(",")}
json.dump(sorted(steps), open(f, "w"))
PY
strip_paths() {    # the capturing machine's paths out of a capture's stamps
    "$PYTHON" - "$1" <<'PY'
import glob, json, sys
for f in glob.glob(f"{sys.argv[1]}/**/capture_config.json", recursive=True):
    c = json.load(open(f))
    for sec in (v for v in c.values() if isinstance(v, dict)):
        for k in [k for k, v in sec.items() if isinstance(v, str) and (v.startswith("/") or k == "pythonpath")]:
            del sec[k]
    open(f, "w").write(json.dumps(c, indent=2))
PY
}
strip_paths "$GR"
PLANNER="$(git log -1 --abbrev=7 --format=%h -- python/perseus/plan)"
PLANNER="$PLANNER (python/perseus/plan's last commit; PLAN_PLACER=ilp)"
SOLVER="scipy $("$PYTHON" -c 'import scipy; print(scipy.__version__)') milp (HiGHS)"
printf '%s\n' "graph=$GR" \
    "tool=speedrun/make_prefill_plan.sh (make_plan.sh blocks + argmax, the tail as make_tail_plan.sh)" \
    "planner=$PLANNER" "recipe=$RECIPE" "runtime=$RUNTIME $SERVER_RUNTIME $REPLAY" "configs=$CONFIGS" "solver=$SOLVER" \
    "pcmm_calib=the largest |y| per linear over the rows $ROWS (the PCMM band shifts)" \
    "rows=the capture's magnitudes are the max over the OpenWebText pool rows $ROWS (merge_mags.py)" \
    > "bootstrap_placements/$OUT/PLAN_CMD.txt"
# the served plan, ${OUT}_cm: CutMax's scaled normalisers on stage 13 (speedrun/make_cutmax_plan.sh) from a capture of
# this runtime with gpt2_base_n32_pf_cm, its stage 13 sized on the logit rows of graphs/gpt2_decode_python_n32_cm
CMC=configs/model/approximation/gpt2_base_n32_pf_cm/configs.json CMG=${GR}_cm
if [ ! -d "$CMG/block_13" ]; then
    rm -rf "$CMG" "$CMG.part"
    PF_PCMM_CALIB_IN="$GR/pcmm_calib.json" "$PYTHON" -m perseus.examples.gpt2.run_prefill \
        --tokens 128 --fake --tail --capture "$CMG.part" --configs "$CMC" $SETS
    ROWS="$(mktemp -d)/rows.npy"
    "$PYTHON" scripts/utils/cutmax_rows.py graphs/gpt2_decode_python_n32_cm/cutmax_rows.json "$ROWS"
    "$PYTHON" -m perseus.examples.gpt2.capture_mags "$CMG.part" --argmax --rows "$ROWS" --blocks 13 \
        --plan "bootstrap_placements/$OUT" --configs "$CMC"
    mv "$CMG.part" "$CMG"
fi
if [ ! -e "$ST/cm" ]; then
    TRACK=prefill GR="$CMG" bash speedrun/make_cutmax_plan.sh "$OUT" "${OUT}_cm"
    touch "$ST/cm"
fi
"$PYTHON" - "$GR" "$CMG" "bootstrap_placements/${OUT}_cm/plan_rot_steps.json" <<'PY'
import json, sys
g, cm, f = sys.argv[1:]      # stages 0-12 from the plan's capture, CutMax's stage 13 from its own
steps = set()
for b in range(14):
    for n in json.load(open(f"{cm if b == 13 else g}/block_{b}/graph.json"))["nodes"]:
        steps |= {int(v) for x in n["inputs"] if x.startswith("rot(") for v in x[4:-1].split(",")}
json.dump(sorted(steps), open(f, "w"))
PY
strip_paths "$CMG"
# the served plan, ${OUT}rd: $OUT's program in every block (PLAN_KEEP_PLACEMENTS_EXACT: its placements, fired hints
# and realize anchors, or the block stops; nothing is left to choose, so the min-cut placer and no solver) and the
# raise-drop search on top, which moves no refresh and changes no block output: block 11 exits as $OUT's does, so stages
# 12 and 13 are $OUT's
RD_RECIPE="PLAN_KEEP_PLACEMENTS=bootstrap_placements/$OUT PLAN_KEEP_PLACEMENTS_EXACT=1 PLAN_PLACER=min_cut \
PLAN_RAISE_DROP_MAX=13 PLAN_RAISE_SEARCH=1 PLAN_RAISE_ROUTES=0,512,1 PLAN_RAISE_DROP_SET=1,2,3,4,5 \
PLAN_RAISE_DROP_BASE=512:1"
TMP="$(mktemp -d graphs/_blocks_XXXX)"
for b in $(seq 0 11); do ln -s "$PWD/$GR/block_$b" "$TMP/block_$b"; done
cp "$GR/capture_env.json" "$TMP/"
( unset MAX_LEVEL BTS_LEVEL SRC_LEVEL CACHE_READ_LEVEL PLAN_LEVEL_UNIT PLAN_CF_MAX PLAN_NO_PRESCALE PLAN_ACC_CHAIN \
      PLAN_MAG_SAFETY PLAN_SPARSE_SLOTS PLAN_SPARSE_BTS_OUT PLAN_HINT_ENV_VETO PLAN_PRUNE PLAN_REAL_ROUTE \
      PLAN_RAISE_DROP_SET PLAN_HINT_VETO_STEPS PLAN_REFRESH_ENV_CAP PLAN_REFRESH_ENV_CAP_STEPS PLAN_SHARED_REALIZE \
      PLAN_REFRESH_ENV_CAP_ROUTES PLAN_RAISE_DROP_MAX PLAN_RAISE_ROUTES PLAN_SLIM_CUBIC PLAN_RAISE_LANDING_MAX \
      PLAN_STC_FIRST_ENTRY PLAN_STC_FIRST_FALLBACK PLAN_MAG_SAFETY_STEPS
  env $RECIPE $RD_RECIPE PLAN_RESUME="${RESUME:-0}" bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$RD" blocks )
rm -rf "$TMP"
"$PYTHON" - "bootstrap_placements/$OUT" "bootstrap_placements/$RD" <<'PY'
import json, sys
a, b = (json.load(open(f"{d}/block_11_placement.json"))["summary"] for d in sys.argv[1:])
if (a["exit_level"], a.get("exit_deg")) != (b["exit_level"], b.get("exit_deg")):
    sys.exit(f"block 11 exits at {b['exit_level']}:{b.get('exit_deg')}, the base's stages 12-13 enter at "
             f"{a['exit_level']}:{a.get('exit_deg')}")
PY
for f in block_12_placement.json block_13_placement.json pcmm_calib.json plan_rot_steps.json; do
    cp "bootstrap_placements/$OUT/$f" "bootstrap_placements/$RD/"
done
# $OUT's recipe file with this pass's tool, placer and recipe, and the raise variants the plan may use (a validity
# list: the session builds the drops its programs use)
TOOL="speedrun/make_prefill_plan.sh (make_plan.sh blocks on $OUT's program; stages 12 and 13 are $OUT's)"
RD_PLANNER="${PLANNER%%(*}(python/perseus/plan's last commit; PLAN_PLACER=min_cut on $OUT's program)"
sed -e "s|^tool=.*|tool=$TOOL|" -e "s|^planner=.*|planner=$RD_PLANNER|" -e "/^recipe=/s|\$| $RD_RECIPE|" \
    -e "/^runtime=/s|FIDESLIB_BTS_RAISE_DROPS=1,3,5|FIDESLIB_BTS_RAISE_DROPS=1,2,3,4,5|" \
    -e "/^solver=/s|\$| ($OUT's)|" "bootstrap_placements/$OUT/PLAN_CMD.txt" > "bootstrap_placements/$RD/PLAN_CMD.txt"
if [ ! -e "$ST/rd_cm" ]; then
    TRACK=prefill GR="$CMG" bash speedrun/make_cutmax_plan.sh "$RD" "${RD}_cm" "$CMC"
    touch "$ST/rd_cm"
fi
cp "bootstrap_placements/${OUT}_cm/plan_rot_steps.json" "bootstrap_placements/${RD}_cm/"
rm -rf "$ST"
ls "bootstrap_placements/$OUT" "bootstrap_placements/${OUT}_cm" \
    "bootstrap_placements/$RD" "bootstrap_placements/${RD}_cm"
