#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/.cache/pip"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -m venv .venv
if [[ "$(uname -s)" == "Darwin" ]]; then
    .venv/bin/python -m pip install 'torch==2.5.1'
else
    .venv/bin/python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt -c constraints.txt
.venv/bin/python -m pip check
if [[ ! -f .env ]]; then cp .env.example .env; fi
