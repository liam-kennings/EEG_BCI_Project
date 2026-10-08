#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
MRCP_SOURCE="$PROJECT_DIR/artifacts/emg_supervised_transfer/mrcp_eeg_emg_source_features.npz"
WAY_SOURCE="$PROJECT_DIR/artifacts/way_eeg_gal_transfer/way_eeg_gal_source_features.npz"
TARGET_CACHE="$PROJECT_DIR/artifacts/emg_supervised_transfer/pure_eeg_target_features.npz"
OUTPUT_DIR="$PROJECT_DIR/artifacts/multisource_privileged_ensemble"

cd "$PROJECT_DIR"

for required in "$MRCP_SOURCE" "$WAY_SOURCE" "$TARGET_CACHE"; do
  if [[ ! -f "$required" ]]; then
    echo "Required standardized cache is missing:"
    echo "  $required"
    echo "Run the MRCP and WAY privileged-transfer workflows first."
    exit 1
  fi
done

python3 scripts/36_multisource_privileged_ensemble.py \
  --mrcp-source-cache "$MRCP_SOURCE" \
  --way-source-cache "$WAY_SOURCE" \
  --target-cache "$TARGET_CACHE" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "Multi-source EEG-only ensemble experiment complete."
echo "Report: $OUTPUT_DIR/MULTISOURCE_PRIVILEGED_ENSEMBLE_REPORT.md"
