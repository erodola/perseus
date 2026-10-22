#!/bin/bash
# The speedrun server on perseus-core with this checkout's toolchain (scripts/local_env.sh); arguments go to
# speedrun/server.py.
set -eo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
source scripts/local_env.sh
exec "$PYTHON" -m speedrun.server "$@"
