#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-deepchem-da}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/environment.yml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists."
else
  "$SOLVER" env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

echo "Conda setup complete. Activate with: conda activate \"$ENV_NAME\""
