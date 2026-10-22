#!/bin/bash
# GPT-2 encrypted decode up to a 512-token context: 512 positions with the encrypted argmax, teacher-forced on a
# 512-token reference; the number is the e2e of the positions near 512 in `[precise] tok<p>`. Every position is
# recorded on the CPU twin at setup and grouped into program classes (the softmax's sparse route grows with the cache:
# Gp x 512 slots, 1024 from 33 keys up to 8192 from 257).
# The reference (block-0 inputs and per-position logits at T=512):
#   .venv/bin/python scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
#       --pool .cache/pools/openwebtext_gpt2.npy --out .cache/oracle/gpt2_ctx512/all_blocks_io --T 16 512
#   GPU=0 NODE=1 MEMS=1,2 ALL_BLOCKS_IO_DIR=.cache/oracle/gpt2_ctx512/all_blocks_io bash scripts/repro/gpt2_decode_ctx512.sh
source "$(dirname "$0")/common.sh"
export STEPS_T=512
bind $PY -m perseus.examples.gpt2.run_decode --tokens 512 --argmax --check \
    --plan bootstrap_placements/gpt2_decode_ctx512 --device "$GPU" "$@"
