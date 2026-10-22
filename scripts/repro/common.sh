# Shared setup of the result runs: the data paths from the repo root, the GPU (GPU=<index>, the device the run may
# use), and an optional NUMA binding (NODE=host node, MEMS=its memory nodes) so the host side runs next to the GPU.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
source scripts/local_env.sh
: "${GPU:=0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
PY=${PYTHON:-python}
bind() { if [ -n "${NODE:-}" ]; then numactl --cpunodebind="$NODE" --membind="${MEMS:-$NODE}" "$@"; else "$@"; fi; }
