# Shared setup of the table runs: the n32 chain env from the repo root, and an optional NUMA binding (NODE=host node,
# MEMS=its memory nodes) so the host side runs next to the GPU. PYTHON defaults to python.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CHAIN=n32 PYTHONUNBUFFERED=1
source scripts/local_env.sh
: "${GPU:=0}"
PY=${PYTHON:-python}
bind() { if [ -n "${NODE:-}" ]; then numactl --cpunodebind="$NODE" --membind="${MEMS:-$NODE}" "$@"; else "$@"; fi; }
