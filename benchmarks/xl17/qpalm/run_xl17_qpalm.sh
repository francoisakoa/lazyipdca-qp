#!/usr/bin/env bash
set -u

ROOT=/data/projects/IPDCA_QP
KIT=${KIT:-$ROOT/cutest_xl17_qpalm_v1}
XLREF=${XLREF:-$ROOT/CUTEST_C_XL_v275_20260825_141918}
MANIFEST=${MANIFEST:-$KIT/manifest_xl17.csv}
MAX_ITER=${MAX_ITER:-50000}
TIME_LIMIT=${TIME_LIMIT:-36000}
TOL=${TOL:-1e-6}
RESUME=${RESUME:-0}

if [ -z "${OUT:-}" ]; then
  OUT="$ROOT/CUTEST_XL17_QPALM_MAX${MAX_ITER}_$(date +%Y%m%d_%H%M%S)"
fi

export MASTSIF="$XLREF/MASTSIF_XL_LOCAL"
export PYCUTEST_CACHE="$ROOT/pycutest_cache_qpalm_xl17"
mkdir -p "$PYCUTEST_CACHE" "$OUT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ ! -d "$MASTSIF" ]; then
  echo "[FAIL] missing XL local MASTSIF: $MASTSIF"
  exit 2
fi

for f in STNQP1 STNQP2 BLOWEYA BLOWEYB BLOWEYC BLOCKQP1 BLOCKQP2 BLOCKQP3; do
  if [ ! -f "$MASTSIF/$f.SIF" ]; then
    echo "[FAIL] missing $MASTSIF/$f.SIF"
    exit 2
  fi
done

echo "===== QPALM CUTEST XL17 ====="
echo "KIT        = $KIT"
echo "XL MASTSIF = $MASTSIF"
echo "CACHE      = $PYCUTEST_CACHE"
echo "MANIFEST   = $MANIFEST"
echo "OUT        = $OUT"
echo "MAX_ITER   = $MAX_ITER"
echo "TIME_LIMIT = $TIME_LIMIT"
echo "TOL        = $TOL"
echo "RESUME     = $RESUME"
echo

ARGS=(
  python -u "$KIT/bench_xl17.py"
  --manifest "$MANIFEST"
  --outdir "$OUT"
  --solvers qpalm
  --max-iter "$MAX_ITER"
  --time-limit "$TIME_LIMIT"
  --tol "$TOL"
  --pycutest-cache "$PYCUTEST_CACHE"
  --cache-prefix qpalm_xl17
  --no-save-x
)

if [ "$RESUME" = "1" ]; then
  ARGS+=(--resume)
fi

"${ARGS[@]}"
RC=$?

printf '%s\n' "$OUT" > "$ROOT/last_cutest_xl17_qpalm_root.txt"

echo
echo "===== FILES ====="
echo "CSV = $OUT/results_xl17_qpalm.csv"
echo "ROOT saved in $ROOT/last_cutest_xl17_qpalm_root.txt"
echo "[INFO] terminal remains open."

return "$RC" 2>/dev/null || true
