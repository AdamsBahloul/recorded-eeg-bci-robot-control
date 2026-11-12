#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS. Use a separate Linux setup for Raspberry Pi."
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "Warning: expected Apple Silicon arm64, detected: $ARCH"
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 was not found."
  echo "Install Python 3.11 (for example from python.org or Homebrew), then rerun."
  exit 1
fi

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-core.txt

echo
echo "Core environment installed."
echo "Activate with: source .venv/bin/activate"
echo "Optional ML stack later: pip install -r requirements-ml.txt"
echo "Optional API stack later: pip install -r requirements-api.txt"
echo "Dataset download is a separate, explicit step."
