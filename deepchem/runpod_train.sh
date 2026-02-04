#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
if [ ! -d "$VENV_DIR" ]; then
  echo "Missing venv at $VENV_DIR. Run ./runpod_setup.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"

python "$ROOT_DIR/train_model.py" "$@"
