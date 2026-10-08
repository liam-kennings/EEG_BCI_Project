#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

python3 -u scripts/38_way_eeg_ssl_seed_stability.py "$@"

echo ""
echo "Five-seed WAY EEG-only SSL stability experiment complete."
echo "Report: $PROJECT_DIR/artifacts/way_eeg_only_ssl_seed_stability/WAY_EEG_SSL_SEED_STABILITY_REPORT.md"
