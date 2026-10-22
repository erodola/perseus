#!/bin/bash
# make_plan.sh — plan the bootstraps of a Llama capture (CPU only), with the recipe and runtime levers of the shipped
# GPT-2 arm (bootstrap_placements/gpt2_decode_python_n32: SparseB on the sparse route (256 slots: Llama's tH), SPRU on the 1-slot route,
# the exact post-raise scaling, the real dense route, raise drops 1,3,5):
#   bash python/perseus/examples/llama/make_plan.sh graphs/llama31_8b_decode_python_n32 llama31_8b_decode_python_n32
# RUNTIME_EXTRA="KEY=VALUE ..." appends switches the capture ran with to the plan's runtime= line (the run
# applies them), e.g. LLAMA_LINEAR_BOTTOM=1.
# The blocks are 0..n_layers-1 plus the tail (final RMSNorm + LM head); the planner's driver is the GPT-2 port's
# make_plan.sh blocks stage (model-independent: one plan per block_<b>/graph.json).
# The runtime= line carries the levers a capture at HEAD runs with (SM_HRS_PACK .. LADDER_RADIX_BITS; without them the
# contract refuses the plan). No raise drops: a dense-route drop 1 on a SiLU refresh returns garbage (the ctx-512
# recalibration's plan planted one in blocks 20-31; KL ~1e19 from the first token).
#   ctx-512 plan (CFG=configs/model/approximation/llama31_8b_n32_ctx512/configs.json):
#     python -m perseus.examples.llama.run_decode --tokens 1 --kv-offload --set KV_ARENA_GB=56 \
#         --set FIDESLIB_BTS_RAISE_DROPS=1,3,5 --set SM_CHEB_GS=0 --capture $GR --structure-only --configs $CFG
#     python -m perseus.examples.llama.capture_mags $GR --configs $CFG
#     bash python/perseus/examples/llama/make_plan.sh $GR llama31_8b_decode_python_n32_ctx512
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; REPO="$(dirname "$(dirname "$PKG")")"
GRAPH_DIR="${1:?graph dir (block_<b>/graph.json)}"
OUT_NAME="${2:?out name under bootstrap_placements/}"
if [[ -d "$GRAPH_DIR" && ! -d "$REPO/$GRAPH_DIR" ]]; then GRAPH_DIR="$(cd "$GRAPH_DIR" && pwd)"; fi
cd "$REPO"
RECIPE="MAX_LEVEL=50 BTS_LEVEL=34 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20 PLAN_NO_PRESCALE=1 \
PLAN_ACC_CHAIN=n32l PLAN_MAG_SAFETY=2 PLAN_SPARSE_BTS_OUT=1:16,256:24 PLAN_DELIB_LANDING_REMAP=24:14 \
PLAN_HINT_ENV_VETO=1 PLAN_PRUNE=1 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_ENV_RULE=nominal PLAN_RAISE_DROP_SET=1,3,5 \
PLAN_REAL_ROUTE=1 PLAN_HINT_VETO_STEPS=tail PLAN_RAISE_ROUTES=0 PLAN_REFRESH_ENV_CAP=50 PLAN_REFRESH_ENV_CAP_STEPS=tail:48 \
FORBID_STEPS=.qkv,.gp PLAN_STEP_LANDING=softmax:32,softmax_v:32"
RUNTIME="FIDESLIB_BTS_SHIFT=1 FIDESLIB_SPRU=64 FIDESLIB_BTS_RAISE_DROPS=1,3,5 SM_DEN_RECIP=1 FIDESLIB_BTS_REAL=1 \
GELU_FOLD=1 LN_CHEB=1 SM_GS_FIRST=1 SM_PERIODIC=2 SM_FOLD=1 FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32 \
KV_LANES=1 CHEB_BSGS=1 QKV_LANES=1 BTS_MAX_INPUT_LEVEL=50 IMPL_SCOPED_NAMES=1 \
SPARSE_BTS_SLOTS=256,1 FIDESLIB_BTS_SPARSE_B_MAX=1024 SPARSE_LEVEL_BUDGET=3,3 \
SM_HRS_PACK=1 CUTMAX_PAIR=1 IMPL_DEAD_OPS=1 LADDER_RADIX_BITS=3 ${RUNTIME_EXTRA:-}"
# exported values win over the recipe (PLAN_* experiments)
for kv in $RECIPE; do k=${kv%%=*}; [ -n "${!k+x}" ] || export "$kv"; done
# the sparse routes follow from Llama's layout (tH = 256), not from the shell: scripts/local_env.sh exports GPT-2's
# 512,1, which the recipe loop above would keep (LLAMA_PLAN_SPARSE_SLOTS overrides)
export PLAN_SPARSE_SLOTS="${LLAMA_PLAN_SPARSE_SLOTS:-256,1}"
RECIPE="$RECIPE PLAN_SPARSE_SLOTS=$PLAN_SPARSE_SLOTS"
# KV_ENTRY_LEVEL/DEG: the level the K/V cache reads arrive at (a decode cut for a prefill hand-off's cache);
# PLAN_KEEP_PLACEMENTS: a plan dir (under bootstrap_placements/) whose refresh sites every block keeps
for k in KV_ENTRY_LEVEL KV_ENTRY_DEG PLAN_KEEP_PLACEMENTS; do if [ -n "${!k:-}" ]; then RECIPE="$RECIPE $k=${!k}"; fi; done
bash "$PKG/examples/gpt2/make_plan.sh" "$GRAPH_DIR" "$OUT_NAME"
MD5=$(cat "$GRAPH_DIR"/block_*/graph.json | md5sum | cut -d' ' -f1)
{
    echo "graph=$GRAPH_DIR (md5 $MD5)"
    echo "tool=perseus/examples/llama/make_plan.sh"
    echo "recipe=$(for kv in $RECIPE; do k=${kv%%=*}; echo -n "$k=${!k} "; done | sed 's/ $//')"
    echo "runtime=$(echo $RUNTIME)"
} > "bootstrap_placements/$OUT_NAME/PLAN_CMD.txt"
cat "bootstrap_placements/$OUT_NAME/PLAN_CMD.txt"
