#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repository: $ROOT"
echo "Python: $(python --version 2>&1)"
echo "CMake: $(cmake --version | head -1)"
echo "Compiler: $(c++ --version | head -1)"
for v in CUTEST MYARCH MASTSIF SIFDECODE; do printf "%-10s %s\n" "$v" "${!v:-<unset>}"; done
