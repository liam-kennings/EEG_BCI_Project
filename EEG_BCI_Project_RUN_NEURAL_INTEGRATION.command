#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"

exec "$PYTHON_BIN" -u scripts/30_translational_motor_intent_v3.py \
  --physionet-dir files/eegmmidb/1.0.0 \
  --openbmi-dir files/external/openbmi/nm000338 \
  --liu-dir "Liu acute-stroke dataset" \
  --cache-path artifacts/gui_experiments/translational_v3/translational_windows_v3.npz \
  --output-dir artifacts/gui_experiments/translational_v4_neural_integrated \
  --neural-bakeoff-dir artifacts/eeg_decoder_model_bakeoff \
  --neural-seeds 42 43 44 45 46 \
  "$@"
