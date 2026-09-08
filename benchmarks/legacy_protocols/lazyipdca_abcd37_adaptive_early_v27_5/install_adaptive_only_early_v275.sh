#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V271_SOURCE=${V271_SOURCE:-/data/projects/IPDCA_QP/lazyipdca_early_lowrank_takeover_v271_source}
V275_SOURCE=${V275_SOURCE:-/data/projects/IPDCA_QP/lazyipdca_adaptive_early_v275_source}
V271_BIN=${V271_BIN:-$V271_SOURCE/build_cutest_early_lowrank_takeover_v271/lazyipdca_cutest}
EXPECTED_V271_SHA=${EXPECTED_V271_SHA:-9a767dfa7294279fa914dd4d363c7b0190e6fd6a89c94ab5f46afd19bfeb13d4}
BUILD=${BUILD:-$V275_SOURCE/build_cutest_adaptive_early_v275}
CONDA_SH=${CONDA_SH:-/home/fakoa/anaconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-lazydca_atom_oracle}
fail(){ echo "[FAIL] $*" >&2; exit 2; }

source "$CONDA_SH"
conda activate "$ENV_NAME"
[ -x "$V271_BIN" ] || fail "missing v2.7.1 binary"
SHA=$(sha256sum "$V271_BIN" | awk '{print $1}')
[ "$SHA" = "$EXPECTED_V271_SHA" ] || fail "v2.7.1 SHA mismatch: $SHA"
echo "[PASS] exact v2.7.1 parent binary verified"

python -m py_compile "$HERE/apply_adaptive_only_early_v275.py" "$HERE/prepare_v271_stage_v275.py" "$HERE/analyze_abcd37_v275.py"
bash -n "$HERE/install_adaptive_only_early_v275.sh"
bash -n "$HERE/run_abcd37_adaptive_early_v275.sh"

echo '===== 1/4 COPY v2.7.1 SOURCE ====='
python "$HERE/prepare_v271_stage_v275.py" --source "$V271_SOURCE" --stage "$V275_SOURCE"

echo '===== 2/4 PATCH ADAPTIVE-ONLY EARLY TAKEOVER ====='
python "$HERE/apply_adaptive_only_early_v275.py" --root "$V275_SOURCE" --dry-run
python "$HERE/apply_adaptive_only_early_v275.py" --root "$V275_SOURCE"

echo '===== 3/4 SYNTAX + BUILD ====='
g++ -std=c++17 -fsyntax-only -fopenmp -Wall -Wextra -Wpedantic \
  -I"$V275_SOURCE/include" -I"$V275_SOURCE/third_party/qdldl" \
  -I"$CONDA_PREFIX/include" -I"$CONDA_PREFIX/include/eigen3" -I/usr/include/eigen3 \
  "$V275_SOURCE/src/solver.cpp"
rm -rf "$BUILD"
cmake -S "$V275_SOURCE" -B "$BUILD" -DLAZYIPDCA_CUTEST=ON
cmake --build "$BUILD" -j1 --target lazyipdca_cutest
NEW="$BUILD/lazyipdca_cutest"
[ -x "$NEW" ] || fail "v2.7.5 binary missing"
python - "$NEW" <<'PY'
from pathlib import Path
import sys
b=Path(sys.argv[1]).read_bytes()
for t in (b'adaptive EARLY TAKEOVER',b'--legacy-atoms',b'--adaptive-atoms',b'--max-seed-atoms'):
    if t not in b:
        raise SystemExit('[FAIL] missing v2.7.5 runtime/CLI token '+t.decode())
print('[PASS] v2.7.5 runtime and A/B/C/D CLI surface verified')
PY

echo '===== 4/4 IMMUTABILITY + READY ====='
[ "$(sha256sum "$V271_BIN" | awk '{print $1}')" = "$SHA" ] || fail "v2.7.1 parent changed"
NEWSHA=$(sha256sum "$NEW" | awk '{print $1}')
echo "[PASS] v2.7.1 parent binary preserved byte-for-byte"
echo "V275_SOURCE=$V275_SOURCE"
echo "V275_BINARY=$NEW"
echo "V275_SHA256=$NEWSHA"
