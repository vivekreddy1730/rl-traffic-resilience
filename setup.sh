#!/usr/bin/env bash
# setup.sh - one-time setup on Linux / macOS (the Windows script setup.ps1 is the tested path).
#   bash setup.sh
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.11 python3.12 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    case "$v" in 3.10|3.11|3.12) PY="$c"; break;; esac
  fi
done
if [ -z "$PY" ]; then
  echo "SETUP FAILED: need Python 3.10, 3.11 or 3.12 (3.13+ not supported by scikit-learn 1.5.0)."
  echo "FIX: install Python 3.11 (e.g. 'sudo apt install python3.11 python3.11-venv' or from python.org) and retry."
  exit 1
fi
echo "== Using $($PY --version)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
# CPU-only PyTorch wheel keeps the Linux download small (no CUDA libraries needed here).
if [ "$(uname)" = "Linux" ]; then
  .venv/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tests/check_setup.py
echo "SETUP OK  -  activate with:  source .venv/bin/activate"
