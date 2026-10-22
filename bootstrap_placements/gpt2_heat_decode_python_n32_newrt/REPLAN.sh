#!/bin/bash
# HEAT GPT-2 on the new runtime: fake capture under the main decode plan's runtime, plan with its recipe (CPU only).
set -euo pipefail
cd /data02/users/mithrillo/projects/perseus-core-wt/m7c
source tools/agent.sh
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4
OLD=/data02/users/mithrillo/projects/perseus/.cache
A=python/perseus_core/assets
NAME=${NAME:-gpt2_heat_decode_python_n32_newrt}
G=$A/graphs/$NAME
MAIN=$A/bootstrap_placements/gpt2_decode_python_n32
W=$OLD/models/openai-community/gpt2/lm_eval_heat/weights.bin.zip
C=configs/model/approximation/gpt2_heat_n32/configs.json
IO=$OLD/oracle/gpt2_heat/all_blocks_io
export PERSEUS_SLOTVEC_CACHE=0 PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export FAKECAP_LANDINGS=0:32,1:14,256:22,512:24
# HEAT's own op stream (its shipped plan's runtime= switches; the rest legacy) on the new runtime's routes
# (the classic sparse-512 route is not built: SparseB, StC-first dense, SPRU, with the main plan's landing knobs)
HEATP=$A/bootstrap_placements/gpt2_heat_decode_python_n32
ROUTES=${ROUTES:-"FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32 FIDESLIB_BTS_RAISE_DROP=512:1 AUTO_BTS_LEVEL=44 BTS_MAX_INPUT_LEVEL=48"}
RUNTIME="$(grep '^runtime=' $HEATP/PLAN_CMD.txt | cut -d= -f2-) $ROUTES"
LEGACY="SM_PERIODIC=0 SM_FOLD=0 KV_LANES=0 CHEB_BSGS=0 QKV_LANES=0 SM_HRS_PACK=0 CUTMAX_PAIR=0 IMPL_DEAD_OPS=0 LADDER_RADIX_BITS=0 SM_CHEB_GS=0 IMPL_SCOPED_NAMES=0"
RUNTIME="$LEGACY $RUNTIME"
# HEAT's recipe with the SparseB 512 landing (classic 512:34 -> SparseB 512:26, as the main recipe)
RECIPE=${RECIPE:-"$(grep '^recipe=' $A/bootstrap_placements/gpt2_heat_decode_python_n32/PLAN_CMD.txt | cut -d= -f2- | sed 's/512:34/512:26/')"}
SETS=""; for kv in $RUNTIME; do SETS="$SETS --set $kv"; done
rm -rf $G
.venv/bin/python -m perseus_core.examples.gpt2.run_decode --fake --tokens 1 --argmax --capture $G \
    --weights $W --configs $C --io-dir $IO $SETS || true   # exit 1 = the 1-token KL gate vs the 12-layer oracle
[ -d $G/block_13 ] || { echo "capture incomplete"; exit 1; }
rm -rf $A/bootstrap_placements/$NAME
env $RECIPE bash python/perseus_core/examples/gpt2/make_plan.sh $G $NAME
env $RECIPE bash python/perseus_core/examples/gpt2/make_plan.sh $G $NAME argmax
{ echo "graph=graphs/$NAME (fake v2 capture, M7c: HEAT weights under the gpt2_decode_python_n32 runtime)"
  echo "tool=perseus_core/examples/gpt2/make_plan.sh (blocks, argmax)"
  echo "recipe=$RECIPE"
  echo "runtime=$RUNTIME"
  echo "configs=$C" ; } > $A/bootstrap_placements/$NAME/PLAN_CMD.txt
.venv/bin/python -c "import glob,json;print('bts/token', sum(json.load(open(f))['summary']['total_bootstraps'] for f in glob.glob('$A/bootstrap_placements/$NAME/block_*_placement.json') if 'feedback' not in f))"
