#!/bin/bash
# GPT-2, encrypted 128-token prefill, cold protocol (GPT2_PF_WARMUP=1: Llama's). SUMMARY wall=.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/gpt2_prefill128.sh
source "$(dirname "$0")/common.sh"
bind $PY -m examples.gpt2_from_primitives.run_prefill --tokens 128 --plan bootstrap_placements/gpt2_prefill_python_n32_blocked \
    --device "$GPU" "$@"
