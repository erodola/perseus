#!/bin/bash
# BERT-base, encrypted 128-token text (SST-2 head), cold protocol: SUMMARY wall= is model.forward on the first input.
# BERT_WARMUP=1 times a second pass after an untimed one (Llama's protocol). Needs scipy (float64 reference GELU).
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/bert_base.sh
source "$(dirname "$0")/common.sh"
bind $PY -m examples.bert_from_primitives.run_bert --text 128 --plan bootstrap_placements/bert_base_python_n32_cc2 \
    --device "$GPU" "$@"
