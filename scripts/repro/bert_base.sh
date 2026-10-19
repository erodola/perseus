#!/bin/bash
# BERT-base, encrypted 128-token text (SST-2 head): SUMMARY wall= times the pass after an untimed warm-up on another input
# (BERT_WARMUP=0: the cold first pass). Needs scipy (float64 reference GELU).
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/bert_base.sh
source "$(dirname "$0")/common.sh"
bind $PY -m examples.bert_from_primitives.run_bert --text 128 --plan bootstrap_placements/bert_base_sst2_128 \
    --device "$GPU" "$@"
