#!/usr/bin/env bash
# CUTEst 37 benchmark: QPALM + OSQP only.
# Run with: bash run_37.sh
# No `exit` is used here, so the caller's interactive shell remains open.

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
KIT=${KIT:-$ROOT/benchmarks/cutest37/qpalm}
MANIFEST=${MANIFEST:-$ROOT/benchmarks/manifests/cutest37.csv}
OUT=${OUT:-$ROOT/CUTEST_37_QPALM_OSQP_$(date +%Y%m%d_%H%M%S)}
SOLVERS=${SOLVERS:-qpalm,osqp}
MAX_ITER=${MAX_ITER:-100000}
TIME_LIMIT=${TIME_LIMIT:-36000}
TOL=${TOL:-1e-6}
ACTIVE_TOL=${ACTIVE_TOL:-1e-5}
QPALM_INNER_TOL=${QPALM_INNER_TOL:-}
QPALM_INNER_MAX_ITER=${QPALM_INNER_MAX_ITER:-}
RESUME=${RESUME:-0}
OSQP_FORCE=${OSQP_FORCE:-0}
PYCUTEST_CACHE_DIR=${PYCUTEST_CACHE_DIR:-$ROOT/pycutest_cache}

export LC_ALL=C LANG=C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$PYCUTEST_CACHE_DIR"
export PYCUTEST_CACHE="$PYCUTEST_CACHE_DIR"

mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/last_cutest37_qpalm_osqp_root.txt"

ARGS=(
  --manifest "$MANIFEST"
  --outdir "$OUT"
  --solvers "$SOLVERS"
  --max-iter "$MAX_ITER"
  --time-limit "$TIME_LIMIT"
  --tol "$TOL"
  --active-tol "$ACTIVE_TOL"
  --pycutest-cache "$PYCUTEST_CACHE_DIR"
)

[ -n "$QPALM_INNER_TOL" ] && ARGS+=(--qpalm-inner-tol "$QPALM_INNER_TOL")
[ -n "$QPALM_INNER_MAX_ITER" ] && ARGS+=(--qpalm-inner-max-iter "$QPALM_INNER_MAX_ITER")

[ "$RESUME" = 1 ] && ARGS+=(--resume)
[ "$OSQP_FORCE" = 1 ] && ARGS+=(--osqp-force)

python "$KIT/preflight.py" | tee "$OUT/preflight.log"
PREFLIGHT_RC=${PIPESTATUS[0]}

if [ "$PREFLIGHT_RC" -ne 0 ]; then
  echo "[FAIL] preflight failed (rc=$PREFLIGHT_RC); campaign not launched."
else
  echo
  echo "===== CAMPAIGN ====="
  echo "ROOT                  = $ROOT"
  echo "KIT                   = $KIT"
  echo "MANIFEST              = $MANIFEST"
  echo "OUT                   = $OUT"
  echo "SOLVERS               = $SOLVERS"
  echo "MAX_ITER              = $MAX_ITER"
  echo "TIME_LIMIT            = $TIME_LIMIT"
  echo "TOL                   = $TOL"
  echo "PYCUTEST_CACHE_DIR     = $PYCUTEST_CACHE_DIR"
  echo "QPALM_INNER_TOL       = ${QPALM_INNER_TOL:-native}"
  echo "QPALM_INNER_MAX_ITER  = ${QPALM_INNER_MAX_ITER:-native}"
  echo "OSQP_FORCE            = $OSQP_FORCE"
  echo

  python "$KIT/bench37.py" "${ARGS[@]}" |& tee "$OUT/campaign.log"

  echo
  echo "===== FILES ====="
  echo "CSV = $OUT/results_37_qpalm_osqp.csv"
  echo "LOG = $OUT/campaign.log"
  echo "ROOT saved in $ROOT/last_cutest37_qpalm_osqp_root.txt"
fi

echo "[INFO] terminal remains open."
