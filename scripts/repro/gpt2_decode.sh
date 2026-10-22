#!/bin/bash
# GPT-2 encrypted decode, 16 tokens with the encrypted argmax: the number is the steady-token (pos >= 1) e2e median of
# `[precise] steady` (the default plan, perseus/examples/defaults.py).
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/gpt2_decode.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.gpt2.run_decode --tokens 16 --argmax --check --device "$GPU" "$@"
