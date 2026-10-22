#!/bin/bash
# Record 0's speedrun server with this checkout's toolchain (scripts/local_env.sh); arguments go to speedrun/server.py.
set -eo pipefail
cd "$(dirname "$0")/.."
export CHAIN=n32 TMPDIR=/tmp
source scripts/local_env.sh
exec "$PYTHON" -m speedrun.server "$@"
