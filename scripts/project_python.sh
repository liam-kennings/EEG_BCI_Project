# Shared interpreter selection for the project launchers (bash and zsh).
# Source this after changing into the project directory.
#
# Order: $EEG_BCI_PYTHON, then the project .venv created by
# EEG_BCI_Project_SETUP_ENV.command, then the framework Python 3.12 that holds
# the pinned scikit-learn 1.6.1 / PyTorch stack, then python3 on PATH.

if [ -n "${EEG_BCI_PYTHON:-}" ]; then
  PYTHON_BIN="$EEG_BCI_PYTHON"
elif [ -x "$PWD/.venv/bin/python" ]; then
  PYTHON_BIN="$PWD/.venv/bin/python"
elif [ -x "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3" ]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PWD/.numba_cache}"
