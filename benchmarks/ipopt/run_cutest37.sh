#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=${OUT:-"$ROOT/results/runs/ipopt37_$(date +%Y%m%d_%H%M%S)"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$ROOT/benchmarks/ipopt/run_cutest_ipopt.py" --panel 37 --outdir "$OUT" --tol 1e-8 --max-iter 3000 --time-limit 36000 --resume
