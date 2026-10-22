#!/bin/bash
# run_notebooks.sh — execute one notebook headless.
#   NB=setup|custom|client_server bash scripts/run_notebooks.sh
# Outputs are written back into the notebook; a cell error fails the run ([notebook] FAIL).
set -e -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source scripts/local_env.sh
export PYTHONFAULTHANDLER=1 HYDRA_FULL_ERROR=1
NB="${NB:?set NB=setup|custom|client_server}"
case "$NB" in
  setup)         NBPATH=notebooks/setup_artifacts.ipynb ;;
  custom)        NBPATH=notebooks/custom_encrypted_model.ipynb ;;
  client_server) NBPATH=notebooks/client_server.ipynb ;;
  *) echo "unknown NB $NB"; exit 1 ;;
esac
echo "[nb] $NB -> $NBPATH"
NB_PATH="$NBPATH" exec "$PYTHON" scripts/utils/run_notebook.py
