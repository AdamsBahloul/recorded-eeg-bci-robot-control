#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_ARCH="$(uname -m)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [[ "${DEVICE_ARCH}" != "aarch64" ]]; then
  echo "Unsupported Pi architecture: ${DEVICE_ARCH}. This pinned runtime requires 64-bit ARM (aarch64)." >&2
  echo "Inspect the Pi OS first; do not attempt an unpinned source build on the 1 GB Pi 3 B+." >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.11 is required; not found: ${PYTHON_BIN}" >&2
  exit 2
fi
"${PYTHON_BIN}" - <<'PY'
import platform, sys
assert sys.version_info[:2] == (3, 11), sys.version
assert platform.machine() == "aarch64", platform.machine()
print("Pi runtime target verified:", platform.platform(), sys.version.split()[0])
PY

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-pi}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --only-binary=:all: -r "${ROOT_DIR}/pi/requirements-pi.txt"
"${VENV_DIR}/bin/python" -m pip check
PYTHONPATH="${ROOT_DIR}" "${VENV_DIR}/bin/python" - "${ROOT_DIR}" <<'PY'
from pathlib import Path
import sys
from embedded.deployment import load_deployment_model

root = Path(sys.argv[1])
model, manifest = load_deployment_model(
    root / "models/deployment/csp_lda.joblib",
    root / "models/deployment/manifest.json",
)
print("Pi model verified:", manifest["model_version"], "classes=", model.classes_.tolist())
PY
echo "Pi dependencies and deployment model verified in ${VENV_DIR}."
