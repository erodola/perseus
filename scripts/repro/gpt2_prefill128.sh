#!/bin/bash
# GPT-2, encrypted 128-token prefill: SUMMARY wall= times the pass after an untimed warm-up on another prompt
# (GPT2_PF_WARMUP=0: the cold first pass).
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/gpt2_prefill128.sh
source "$(dirname "$0")/common.sh"
bind $PY -m examples.gpt2_from_primitives.run_prefill --tokens 128 --plan bootstrap_placements/gpt2_prefill128 \
    --device "$GPU" "$@"
