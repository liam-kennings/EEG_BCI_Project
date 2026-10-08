#!/bin/zsh
set -euo pipefail

# Reproduce the current V7 result (graph/LDA-enhanced binary intent gate)
# from the cached windows and the five trained neural seeds. Uses quality-gate
# scoring version 1 so the recorded V7 numbers reproduce exactly; drop
# --legacy-quality-gate to run with the fixed gate. Writes the report,
# participant-level metrics, and per-window predictions to
# artifacts/gui_experiments/translational_v7_graph_lda_reproduction.

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh
export MPLBACKEND="Agg"

exec "$PYTHON_BIN" -u scripts/30_translational_motor_intent_v3.py \
  --physionet-dir files/eegmmidb/1.0.0 \
  --openbmi-dir files/external/openbmi/nm000338 \
  --liu-dir "Liu acute-stroke dataset" \
  --cache-path artifacts/gui_experiments/translational_v5_patient_adaptive/translational_windows_v3.npz \
  --output-dir artifacts/gui_experiments/translational_v7_graph_lda_reproduction \
  --neural-bakeoff-dir artifacts/gui_experiments/eeg_decoder_model_bakeoff \
  --neural-seeds 42 43 44 45 46 \
  --neural-architectures mrcp_fbcnet shallow_convnet \
  --neural-device auto \
  --target-intent-sensitivity 0.7 \
  --temporal-confirmations 3 \
  --temporal-refractory-seconds 2.0 \
  --bootstrap-iterations 1000 \
  --auxiliary-direction \
  --legacy-quality-gate \
  "$@"
