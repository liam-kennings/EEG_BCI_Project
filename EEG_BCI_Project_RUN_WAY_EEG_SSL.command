#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
WAY_DIR="$PROJECT_DIR/WAY-EEG-GAL"
WAY_CACHE="$PROJECT_DIR/artifacts/way_eeg_only_ssl/way_unlabeled_eeg_windows.npz"
TARGET_RAW="$PROJECT_DIR/../artifacts/gui_experiments/translational_v3/translational_windows_v3.npz"
TARGET_FEATURES="$PROJECT_DIR/artifacts/emg_supervised_transfer/pure_eeg_target_features.npz"
COMPARISON_MODEL="$PROJECT_DIR/artifacts/multisource_privileged_ensemble/multisource_eeg_intent_ensemble.joblib"
OUTPUT_DIR="$PROJECT_DIR/artifacts/way_eeg_only_ssl"

cd "$PROJECT_DIR"

for required in "$TARGET_RAW" "$TARGET_FEATURES" "$COMPARISON_MODEL"; do
  if [[ ! -f "$required" ]]; then
    echo "Required target experiment artifact is missing:"
    echo "  $required"
    exit 1
  fi
done

python3 -u scripts/37_way_eeg_only_ssl_transfer.py \
  --way-dir "$WAY_DIR" \
  --way-cache "$WAY_CACHE" \
  --target-raw-cache "$TARGET_RAW" \
  --target-feature-cache "$TARGET_FEATURES" \
  --comparison-model "$COMPARISON_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo ""
echo "WAY EEG-only self-supervised experiment complete."
echo "Report: $OUTPUT_DIR/WAY_EEG_ONLY_SSL_TRANSFER_REPORT.md"
