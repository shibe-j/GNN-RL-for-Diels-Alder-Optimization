#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-deepchem-da}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Missing conda env $ENV_NAME. Run ./runpod_setup_conda.sh first."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python "$ROOT_DIR/train_model.py" "$@"
