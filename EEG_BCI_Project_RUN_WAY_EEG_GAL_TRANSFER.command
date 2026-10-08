#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
SOURCE_DIR="$PROJECT_DIR/WAY-EEG-GAL"
OUTPUT_DIR="$PROJECT_DIR/artifacts/way_eeg_gal_transfer"
TARGET_CACHE="$PROJECT_DIR/artifacts/emg_supervised_transfer/pure_eeg_target_features.npz"

cd "$PROJECT_DIR"

if [[ ! -f "$SOURCE_DIR/.way_eeg_gal_ready" ]]; then
  echo "WAY-EEG-GAL is not marked ready. Run EEG_BCI_Project_DOWNLOAD_WAY_EEG_GAL.command first."
  exit 1
fi

if [[ ! -f "$TARGET_CACHE" ]]; then
  echo "The shared pure-EEG target cache is missing:"
  echo "  $TARGET_CACHE"
  echo "Run EEG_BCI_Project_RUN_EMG_SUPERVISED_TRANSFER.command once to prepare it."
  exit 1
fi

python3 scripts/35_prepare_way_eeg_gal_transfer.py \
  --data-dir "$SOURCE_DIR" \
  --target-cache "$TARGET_CACHE" \
  --output-dir "$OUTPUT_DIR"

python3 scripts/32_emg_supervised_eeg_transfer.py \
  --source-cache "$OUTPUT_DIR/way_eeg_gal_source_features.npz" \
  --target-cache "$TARGET_CACHE" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "WAY-assisted EEG-only decoder experiment complete."
echo "Report: $OUTPUT_DIR/EMG_SUPERVISED_EEG_TRANSFER_REPORT.md"
