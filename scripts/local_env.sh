#!/bin/bash
# local_env.sh — where the examples find their data, on one machine.
#   source scripts/local_env.sh
# Every value is a default: anything already exported by the caller wins. The CKKS parameters and the runtime switches
# come from the plan (its PLAN_CMD.txt runtime= line) and the Config presets, not from the environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-$REPO/.cache/tmp}"; mkdir -p "$TMPDIR"

# ---- host resources: one FHE process at a time
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# ---- data: model weights and the reference outputs (README: "Pipeline")
export PERSEUS_DATA="${PERSEUS_DATA:-$REPO/.cache}"
export WEIGHTS_PATH="${WEIGHTS_PATH:-$PERSEUS_DATA/models/openai-community/gpt2/classic/weights.bin.zip}"
export ALL_BLOCKS_IO_DIR="${ALL_BLOCKS_IO_DIR:-$PERSEUS_DATA/oracle/gpt2/all_blocks_io}"
