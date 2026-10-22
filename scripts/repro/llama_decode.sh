#!/bin/bash
# Llama-3.1-8B, encrypted decode of 16 tokens (weights streamed from pinned host memory): the number is the median
# token time.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/llama_decode.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.llama.run_decode --tokens 16 --device "$GPU" "$@"
