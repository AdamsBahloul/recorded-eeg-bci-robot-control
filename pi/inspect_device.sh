#!/usr/bin/env bash
set -euo pipefail

echo "=== Raspberry Pi OS ==="
cat /etc/os-release
echo
echo "=== Architecture ==="
uname -m
getconf LONG_BIT
echo
echo "=== Python 3.11 ==="
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 -c 'import platform,sys; print(sys.version); print(platform.machine())'
else
  echo "python3.11 is not installed"
fi
echo
echo "=== Memory ==="
free -h
echo
echo "=== Network ==="
hostname -I
