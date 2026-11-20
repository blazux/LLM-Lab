#!/usr/bin/env bash
# Install the application in its own environment; never modify system Python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${LLMLAB_DEVICE:-auto}"
VENV="${LLMLAB_VENV:-$ROOT/.venv}"
PYTHON="${LLMLAB_PYTHON:-python3}"
while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) echo 'Usage: bash scripts/bootstrap.sh [--device auto|cpu|cuda] [--venv PATH] [--python PYTHON]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$DEVICE" in auto|cpu|cuda) ;; *) echo 'Device must be auto, cpu or cuda.' >&2; exit 2 ;; esac
if [[ "$DEVICE" == auto ]]; then
  DEVICE=cpu
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then DEVICE=cuda; fi
fi
if [[ "$DEVICE" == cuda ]]; then CHANNEL=cu130; else CHANNEL=cpu; fi
"$PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Use Python 3.11, 3.12 or 3.13."'
if [[ ! -x "$VENV/bin/python" ]]; then "$PYTHON" -m venv "$VENV"; fi
INSTALLER=("$VENV/bin/python" -m pip)
# uv also works with minimal venvs that do not bundle pip.
if command -v uv >/dev/null 2>&1; then INSTALLER=(uv pip --python "$VENV/bin/python"); fi
if [[ "${INSTALLER[0]}" == uv ]]; then
  uv pip install --python "$VENV/bin/python" "torch==2.13.0+$CHANNEL" --index-url "https://download.pytorch.org/whl/$CHANNEL"
  uv pip install --python "$VENV/bin/python" -c "$ROOT/constraints.txt" -r "$ROOT/requirements.txt" -r "$ROOT/gui/backend/requirements.txt"
else
  "${INSTALLER[@]}" install "torch==2.13.0+$CHANNEL" --index-url "https://download.pytorch.org/whl/$CHANNEL"
  "${INSTALLER[@]}" install -c "$ROOT/constraints.txt" -r "$ROOT/requirements.txt" -r "$ROOT/gui/backend/requirements.txt"
fi
"$VENV/bin/python" "$ROOT/scripts/check_environment.py" --device "$DEVICE"
echo "Environment ready: $VENV ($DEVICE)"
