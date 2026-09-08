#!/usr/bin/env bash
set -u
ROOT=${ROOT:-/data/projects/IPDCA_QP}
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
OUT=${OUT:-$ROOT/LCP_XL_QPALM_OSQP_$(date +%Y%m%d_%H%M%S)}
MAX_ITER=${MAX_ITER:-50000}; TIME_LIMIT=${TIME_LIMIT:-36000}; TOL=${TOL:-1e-6}; SOLVERS=${SOLVERS:-qpalm,osqp}; RESUME=${RESUME:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ARGS=(--outdir "$OUT" --solvers "$SOLVERS" --max-iter "$MAX_ITER" --time-limit "$TIME_LIMIT" --tol "$TOL")
[[ "$RESUME" == 1 ]] && ARGS+=(--resume)
echo "===== LCP XL CAMPAIGN ====="; echo "KIT=$KIT"; echo "OUT=$OUT"; echo "MAX_ITER=$MAX_ITER TIME_LIMIT=$TIME_LIMIT TOL=$TOL SOLVERS=$SOLVERS RESUME=$RESUME"
python "$KIT/preflight.py" || true
python "$KIT/lcp_xl_bench.py" "${ARGS[@]}" |& tee "$OUT/campaign.log"
echo "$OUT" > "$ROOT/last_lcp_xl_qpalm_osqp_root.txt"
echo "[INFO] ROOT saved in $ROOT/last_lcp_xl_qpalm_osqp_root.txt"
