#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"

SOURCE_CACHE="$PROJECT_DIR/artifacts/emg_supervised_transfer/mrcp_eeg_emg_source_features.npz"
TARGET_CACHE="$PROJECT_DIR/artifacts/emg_supervised_transfer/pure_eeg_target_features.npz"

if [[ $# -eq 0 ]]; then
  if [[ ! -f "$SOURCE_CACHE" || ! -f "$TARGET_CACHE" ]]; then
    "$PYTHON_BIN" scripts/33_prepare_mrcp_eeg_emg_transfer.py
  fi
  exec "$PYTHON_BIN" scripts/32_emg_supervised_eeg_transfer.py \
    --source-cache "$SOURCE_CACHE" \
    --target-cache "$TARGET_CACHE"
fi

exec "$PYTHON_BIN" scripts/32_emg_supervised_eeg_transfer.py "$@"
