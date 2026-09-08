#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; BIN=${BIN:-$ROOT/build-cutest/lazyipdca_cutest}; MANIFEST=${MANIFEST:-$ROOT/benchmarks/manifests/cutest37.csv}; OUT=${OUT:-$ROOT/results/runs/scalar37_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COMMON=(--manifest "$MANIFEST" --init 2 --backend ldl --max-iter 3000 --time-limit 36000 --eps 1e-6 --rho-init auto --rho-init-rel-margin 3.0 --rho-init-abs-margin 1e-8 --rho-rel-margin 0.10 --rho-curvature-tol 1e-8 --rho-dyn-floor-rel-margin 1e-6 --rho-trial-every 2 --rho-decay 0.5 --reject-streak-rho-bump 50 --rho-dyn-reject-growth 2.0 --mehrotra --dca-stop --dca-stall-tol-rel 1e-10 --dca-stall-iters 5 --dca-monotone-tol-abs 1e-6 --bound-push 0 --bp-push-kappa 1e-4 --bp-margin-tol 1e-4 --bp-adaptive --escapes 3 --sigma-max 1e14 --model-solved-tol 1e-12 --dca-momentum original --dca-acceptance monotone --no-momentum-complete --verbose 1)
mapfile -t TAGS < <(python "$ROOT/benchmarks/cutest37/tags_from_manifest.py" "$MANIFEST")
for mode in const dyn; do for tag in "${TAGS[@]}"; do "$BIN" "${COMMON[@]}" --regularization-mode "$mode" --problem "$tag" --outdir "$OUT/$mode/$tag" > "$OUT/${mode}_${tag}.log" 2>&1 || true; done; done
echo "$OUT"
