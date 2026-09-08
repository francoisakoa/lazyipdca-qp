#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m pip install -U pip
python -m pip install "$ROOT"
python -c "import lazyipdca; print('lazyipdca Python module imported successfully')"
