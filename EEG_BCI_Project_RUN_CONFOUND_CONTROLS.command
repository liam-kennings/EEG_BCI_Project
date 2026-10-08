#!/bin/zsh
set -euo pipefail

# Rebuild windows under cue/trial-phase controls (pre-cue rest, later intent
# windows, causal filters) and score transparent decoders on the V7
# evaluation participants. First run builds a ~100 MB feature cache.

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh
export MPLBACKEND="Agg"

exec "$PYTHON_BIN" -u scripts/54_confound_control_benchmark.py "$@"
