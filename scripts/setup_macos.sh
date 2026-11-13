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

PYTHON311="$(command -v python3.11 || true)"
if [[ -z "$PYTHON311" && -x /opt/homebrew/bin/python3.11 ]]; then
  PYTHON311="/opt/homebrew/bin/python3.11"
fi

if [[ -z "$PYTHON311" ]]; then
  echo "Python 3.11 was not found."
  echo "Install Python 3.11, then rerun."
  exit 1
fi

"$PYTHON311" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-m1-lock.txt

echo
echo "Milestone 1 environment installed."
echo "Activate with: source .venv/bin/activate"
echo "MuJoCo, classifier, and API dependencies are deferred to their milestones."
echo "Dataset download is a separate, explicit step."
