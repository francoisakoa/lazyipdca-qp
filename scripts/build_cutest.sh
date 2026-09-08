#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${CUTEST:?source the CUTEst env.sh first}"
: "${MYARCH:?source the CUTEst env.sh first}"
BUILD=${BUILD:-$ROOT/build-cutest}
rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLAZYIPDCA_TESTS=ON -DLAZYIPDCA_PYTHON=OFF -DLAZYIPDCA_CUTEST=ON
cmake --build "$BUILD" -j"${JOBS:-1}" --target lazyipdca_cutest
echo "Binary: $BUILD/lazyipdca_cutest"
