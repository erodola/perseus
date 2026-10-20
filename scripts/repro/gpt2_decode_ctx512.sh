#!/bin/bash
# GPT-2 encrypted decode up to a 512-token context (the ROSETTA comparison): 512 positions with the encrypted argmax,
# teacher-forced on a 512-token reference; the number is the e2e median of the positions near 512 in `[precise]`.
# The reference is the table's own pool and offset at T=512 (its first 16 positions are the 16-token reference):
#   .venv/bin/python scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
#       --pool .cache/pools/openwebtext_gpt2.npy --out .cache/oracle/gpt2_ctx512/all_blocks_io --T 16 512
# The plan streams the block weights (IMPL_RESIDENT_BLOCKS=0) to make room for the softmax's growing sparse route
# (Gp x 512 slots: 1024 from 33 keys up to 8192 from 257); each class change builds its route (~20 s) at the first
# position past the boundary (32, 64, 128, 256).
#   GPU=0 NODE=1 MEMS=1,2 ALL_BLOCKS_IO_DIR=.cache/oracle/gpt2_ctx512/all_blocks_io bash scripts/repro/gpt2_decode_ctx512.sh
source "$(dirname "$0")/common.sh"
export STEPS_T=512
bind $PY scripts/repro/decode_precise.py --tokens 512 --argmax --plan bootstrap_placements/gpt2_decode_python_n32_ctx512 \
    --device "$GPU" "$@"
