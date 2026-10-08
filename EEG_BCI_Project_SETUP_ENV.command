#!/bin/zsh
set -euo pipefail

# Create the project virtual environment used by every launcher.
# On this Mac the framework Python 3.12 already holds the pinned stack
# (scikit-learn 1.6.1, PyTorch with Metal), so the venv inherits it and only
# installs what differs, currently scipy 1.16.3. Elsewhere it installs the
# full requirements.

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

FRAMEWORK_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

if [[ -x .venv/bin/python ]]; then
  print "Existing environment: $PROJECT_DIR/.venv"
elif [[ -x "$FRAMEWORK_PYTHON" ]]; then
  "$FRAMEWORK_PYTHON" -m venv --system-site-packages .venv
else
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt
if [[ "${1:-}" == "--with-optional" ]]; then
  .venv/bin/python -m pip install -r requirements-optional.txt
fi

.venv/bin/python - <<'PY'
import numpy, scipy, sklearn, mne
import scipy.optimize, scipy.sparse.linalg, scipy.integrate  # fail fast on broken builds
print(f"numpy {numpy.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, mne {mne.__version__}")
try:
    import torch
    print(f"torch {torch.__version__}, mps={torch.backends.mps.is_available()}")
except ImportError:
    print("torch not installed (run with --with-optional for neural decoders)")
PY
print "Environment ready: $PROJECT_DIR/.venv"
