#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

PREFERRED_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
if [[ -n "${EEG_BCI_PYTHON:-}" ]]; then
  PYTHON_BIN="$EEG_BCI_PYTHON"
elif [[ -x "$PREFERRED_PYTHON" ]]; then
  PYTHON_BIN="$PREFERRED_PYTHON"
else
  PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" -u scripts/39_eeg_decoder_model_bakeoff.py "$@"

echo ""
echo "EEG decoder model bakeoff complete."
echo "Report: $PROJECT_DIR/artifacts/eeg_decoder_model_bakeoff/EEG_DECODER_MODEL_BAKEOFF_REPORT.md"
