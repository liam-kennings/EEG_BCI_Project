#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh

exec "$PYTHON_BIN" scripts/34_download_way_eeg_gal.py --extract "$@"
