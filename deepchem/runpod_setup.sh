#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  TORCH_INDEX_URL_VALUE="$TORCH_INDEX_URL"
elif command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX_URL_VALUE="https://download.pytorch.org/whl/cu121"
else
  TORCH_INDEX_URL_VALUE="https://download.pytorch.org/whl/cpu"
fi

python -m pip install --index-url "$TORCH_INDEX_URL_VALUE" torch torchvision torchaudio

TORCH_VERSION="$(python - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0])
PY
)"

CUDA_TAG="$(python - <<'PY'
import torch
cuda_version = torch.version.cuda
if cuda_version:
    print(f"cu{cuda_version.replace('.', '')}")
else:
    print("cpu")
PY
)"

PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f "$PYG_WHL_URL"

python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Setup complete. Activate with: source \"$VENV_DIR/bin/activate\""
