#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"

PREFERRED_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
if [[ -n "${EEG_BCI_PYTHON:-}" ]]; then
  PYTHON_BIN="$EEG_BCI_PYTHON"
elif [[ -x "$PREFERRED_PYTHON" ]]; then
  PYTHON_BIN="$PREFERRED_PYTHON"
else
  PYTHON_BIN="$(command -v python3)"
fi

exec "$PYTHON_BIN" scripts/22_bci_pipeline_gui.py "$@"
