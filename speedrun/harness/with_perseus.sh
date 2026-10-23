#!/bin/bash
# Run a Python script from the Perseus checkout DIR with its toolchain (scripts/local_env.sh): how the client gets its
# crypto backend (harness/perseus_client.py). Paths in ARGS must be absolute.
#   bash harness/with_perseus.sh DIR SCRIPT ARGS...
set -eo pipefail
D=$1; shift
cd "$D"
export CHAIN=n32 TMPDIR=/tmp
source scripts/local_env.sh
export PYTHONPATH="$D:${PYTHONPATH:-}"     # the checkout's speedrun/ package
exec "$PYTHON" "$@"
