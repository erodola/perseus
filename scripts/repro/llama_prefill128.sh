#!/bin/bash
# Llama-3.1-8B, encrypted 128-token prefill. Every switch rides in the plan's runtime= line. The wall (SUMMARY wall=) is the
# timed pass after an untimed warm-up on another window: keys, masks and block-0 setup are excluded.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/llama_prefill128.sh
source "$(dirname "$0")/common.sh"
bind $PY -m examples.llama_from_primitives.run_prefill --plan bootstrap_placements/llama31_8b_prefill_python_n32_pr2s12456e2r0sme2 \
    --tokens 128 --device "$GPU" "$@"
