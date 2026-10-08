#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
source scripts/project_python.sh

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"
export MPLBACKEND="Agg"

exec "$PYTHON_BIN" scripts/51_v7h_mind_benchmark.py \
  --mind-dir MIND \
  --output-dir artifacts/v7h_mind \
  "$@"
