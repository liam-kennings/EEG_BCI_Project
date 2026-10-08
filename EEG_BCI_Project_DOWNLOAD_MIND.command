#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

exec python3 scripts/50_download_mind.py \
  --profile derivatives \
  --accept-license \
  "$@"
