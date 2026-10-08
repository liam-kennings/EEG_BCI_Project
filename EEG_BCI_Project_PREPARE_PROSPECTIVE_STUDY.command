#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"

python3 -u scripts/40_prepare_prospective_controller.py "$@"
python3 -u scripts/41_mrcp_phase_ablation.py
