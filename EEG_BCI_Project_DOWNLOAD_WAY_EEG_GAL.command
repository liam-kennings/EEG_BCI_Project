#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

exec python3 scripts/34_download_way_eeg_gal.py --extract "$@"
