#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=${OUT:-"$ROOT/results/runs/lcp67_ipopt_$(date +%Y%m%d_%H%M%S)"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$ROOT/benchmarks/lcp/run_ipopt_certificate_matched.py" --outdir "$OUT" --resume
