#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"
export MPLBACKEND="Agg"

exec "$PYTHON_BIN" scripts/51_v7h_mind_benchmark.py \
  --mind-dir MIND \
  --output-dir artifacts/v7h_mind \
  "$@"
