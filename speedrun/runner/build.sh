#!/bin/bash
# Build one Perseus commit into a new directory, as the harness runs it: a PR's commit (the one its check names), the
# current record, or the commit a client pins. The clone is standalone (its own .git, nothing shared with yours), at
# COMMIT fetched from REPO (a path or URL; for a PR, GitHub serves its head by SHA), with its submodules (no local
# file:// URLs) and no git hooks. Building runs the commit's own code (build scripts, its Python dependencies), so run
# it where an entry may run: an account without your credentials, or a VM. It needs cargo (rustup), uv and CUDA
# (CUDA_HOME, default /usr/local/cuda-13.2) on the machine. EXPORT=1 also exports GPT-2's weights, for a client's
# build: the server reads the weights from there.
#   [EXPORT=1] bash speedrun/runner/build.sh REPO COMMIT DIR
set -euo pipefail
R=$1 C=$2 D=$3
[ -d "$R" ] && R=$(cd "$R" && pwd)
[[ "$C" =~ ^[0-9a-f]{40}$ ]] || { echo "build.sh: COMMIT must be a full 40-character SHA" >&2; exit 2; }
[ ! -e "$D" ] || { echo "build.sh: $D exists" >&2; exit 2; }
G=(git -c core.hooksPath=/dev/null)
"${G[@]}" init -q "$D"
cd "$D"
"${G[@]}" fetch -q --no-tags --depth=1 "$R" "$C"
"${G[@]}" checkout -q --detach "$C"
"${G[@]}" -c protocol.file.allow=never submodule update -q --init --recursive --depth=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
export PATH="$CUDA_HOME/bin:$PATH"
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -e ".[hf,examples,dev]"
.venv/bin/maturin develop --release
if [ "${EXPORT:-0}" = 1 ]; then
    .venv/bin/perseus-export --model openai-community/gpt2 --out .cache/models/openai-community/gpt2
fi
echo "build.sh: $D at $C"
