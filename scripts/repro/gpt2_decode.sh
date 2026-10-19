#!/bin/bash
# GPT-2 encrypted decode, 16 tokens with the encrypted argmax on the server: the table number is the steady-token
# (pos >= 1) e2e median of `[precise] steady`.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/gpt2_decode.sh
source "$(dirname "$0")/common.sh"
bind $PY scripts/repro/decode_precise.py --tokens 16 --argmax --plan bootstrap_placements/gpt2_decode_python_n32 \
    --device "$GPU" "$@"
