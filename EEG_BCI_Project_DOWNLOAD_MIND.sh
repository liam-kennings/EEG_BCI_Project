#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# The derivatives profile is sufficient for the first V7-H benchmark.
# Running this command confirms acceptance of the dataset's CC BY-NC 4.0 terms.
exec python3 scripts/50_download_mind.py \
  --profile derivatives \
  --accept-license \
  "$@"
