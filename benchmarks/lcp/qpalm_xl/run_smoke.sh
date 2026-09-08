#!/usr/bin/env bash
set -u
ROOT=${ROOT:-/data/projects/IPDCA_QP}
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
OUT=${OUT:-$ROOT/LCP_QPALM_OSQP_SMOKE}
rm -rf "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python "$KIT/lcp_xl_bench.py" --outdir "$OUT" --solvers qpalm,osqp --only LCP7:10000 --only LCP11:10000 --max-iter 50000 --time-limit 36000 --tol 1e-6
