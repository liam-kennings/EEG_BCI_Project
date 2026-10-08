#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh

"$PYTHON_BIN" -u scripts/39_eeg_decoder_model_bakeoff.py "$@"

echo ""
echo "EEG decoder model bakeoff complete."
echo "Report: $PROJECT_DIR/artifacts/eeg_decoder_model_bakeoff/EEG_DECODER_MODEL_BAKEOFF_REPORT.md"
