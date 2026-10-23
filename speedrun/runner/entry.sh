#!/bin/bash
# Run a Perseus entry through the harness on one GPU, as a submission: the server is PERSEUS_DIR's speedrun/serve.sh,
# and the client gets the crypto of CLIENT (default PERSEUS_DIR) for BACKEND (perseus: record 0, FIDESlib;
# perseus-core: record 1, the Rust runtime). The client holds the secret key and the prompts, so CLIENT must be your
# build of the Perseus commit that harness/BACKEND_client.py pins (PERSEUS); run.py refuses any other. The server reads
# GPT-2's weights from WEIGHTS (default: CLIENT's perseus-export), read-only at $WEIGHTS_PATH, and its checkout may link
# (top-level symlinks, its venv) only into what CLIENT links to or into ALLOW (DIR[:DIR...]). NUMA=NODE[:MEMS],
# optional, pins the server to the GPU's NUMA node and caps its CPU time at the node's CPUs. The client checks PROMPTS (default prompts/dev.json) against the
# float64 reference on REF (default: openai-community/gpt2's model.safetensors in $HF_HOME). Runs land in $RUNS
# (default runs/). Memory caps: 160 GB for the server, 64 GB for the client, within the reference box's 256 GB
# (record 1 peaks at 39.6 and 17.7 GB, in Prefill-128).
#   [CLIENT=DIR] bash runner/entry.sh BACKEND PERSEUS_DIR {prefill,decode} GPU [run.py options...]
set -euo pipefail
B=$1; P=$(cd "$2" && pwd); T=$3; G=$4; shift 4
C=$(cd "${CLIENT:-$P}" && pwd)
H=$(cd "$(dirname "$0")" && pwd)
G2="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--openai-community--gpt2"
AL=()
IFS=: read -ra DIRS <<< "${ALLOW:-}"
for d in "${DIRS[@]}"; do AL+=(--allow "$d"); done
exec python3 -I "$H/run.py" --track "$T" --gpu "$G" --runs "${RUNS:-$H/../runs}" --mem 160G --client-mem 64G \
    ${NUMA:+--numa "$NUMA"} --perseus "$C" --backend "$B" --prompts "${PROMPTS:-$H/../prompts/dev.json}" \
    --ref "${REF:-$G2/snapshots/$(cat "$G2/refs/main" 2>/dev/null)/model.safetensors}" \
    --weights "${WEIGHTS:-$C/.cache/models/openai-community/gpt2/classic/weights.bin.zip}" "${AL[@]}" \
    --ro "$P" "$@" -- bash "$P/speedrun/serve.sh" --device 0   # the sandbox's one GPU
