#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD=${BUILD:-$ROOT/build}
rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLAZYIPDCA_TESTS=ON -DLAZYIPDCA_PYTHON=OFF -DLAZYIPDCA_CUTEST=OFF
cmake --build "$BUILD" -j"${JOBS:-1}"
