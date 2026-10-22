#!/bin/bash
# M8 replan recipe of this dir: fake v2 capture (CPU) of the source plan's runtime, then make_plan.sh (blocks).
# graph: graphs/gpt2_prefill_python_n32_pcmmstcf_m8 -> perseus-oracle/fx/dumps/M8/graphs (0 structural diffs vs the old repo's graph)
set -euo pipefail
# (rerunning it rebuilds this dir: keep a copy of this script first)
export NAME=gpt2_prefill_python_n32_pcmmstcf_m8 PLAN=gpt2_prefill_python_n32_pcmmstcf SRC=gpt2_prefill_python_n32_pcmmstcf TOK=64
# --- capture
cd /data02/users/mithrillo/projects/perseus-core-wt/m8
source tools/agent.sh
export OMP_NUM_THREADS=8
OLD=/data02/users/mithrillo/projects/perseus
A=python/perseus_core/assets
G=$A/graphs/$NAME
export CUDA_VISIBLE_DEVICES= PERSEUS_SLOTVEC_CACHE=0 PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1
export WEIGHTS_PATH=$OLD/.cache/models/openai-community/gpt2/classic/weights.bin.zip ALL_BLOCKS_IO_DIR=$OLD/.cache/oracle/gpt2/all_blocks_io
export FAKECAP_LANDINGS=$(.venv/bin/python -c "from perseus_core import config; print(config.preset('n32_gpt2_prefill')['quirks.fakecap_landings'])")
export PF_PCMM_CALIB_IN=$OLD/bootstrap_placements/$PLAN/pcmm_calib.json
SETS=$(.venv/bin/python -c "from perseus_core import config; print(' '.join('--set '+k+'='+v for k,v in config.runtime_env('bootstrap_placements/$PLAN').items()))")
rm -rf $G
/usr/bin/time -v .venv/bin/python -m perseus_core.examples.gpt2.run_prefill --fake --tokens $TOK --capture $G $SETS ${EXTRA:-}
# --- plan
RECIPE=${RECIPE:-"BTS_LEVEL=28 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_LANDING_MAX=42 PLAN_RAISE_ROUTES=0 PLAN_REFRESH_ENV_CAP_ROUTES=0:46 PLAN_SLIM_CUBIC=1 PLAN_STC_FIRST_ENTRY=46 PLAN_STC_FIRST_FALLBACK=34:1"}
rm -rf $A/bootstrap_placements/$NAME
env $RECIPE bash python/perseus_core/examples/gpt2/make_plan.sh graphs/$NAME $NAME
cp $OLD/bootstrap_placements/$SRC/pcmm_calib.json $A/bootstrap_placements/$NAME/
{ echo "graph=graphs/$NAME (fake v2 capture, M8: T=${TOK:-64} under $SRC's runtime + its pcmm_calib.json)"
  echo "tool=perseus_core/examples/gpt2/make_plan.sh (blocks)"
  echo "recipe=$RECIPE"
  grep '^runtime=' $OLD/bootstrap_placements/$SRC/PLAN_CMD.txt
  echo "replanned=M8 on the new Shadow" ; } > $A/bootstrap_placements/$NAME/PLAN_CMD.txt
.venv/bin/python -c "import glob,json;print('bts', sum(json.load(open(f))['summary']['total_bootstraps'] for f in glob.glob('$A/bootstrap_placements/$NAME/block_*_placement.json')))"
