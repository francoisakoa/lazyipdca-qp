#!/usr/bin/env bash
set -euo pipefail

# Controlled OMS ablation: same executable / manifest / common options,
# only the DC regularization mode changes.
# Usage:
#   LAZYIPDCA_CUTEST_BIN=./build/lazyipdca_cutest \
#     tools/run_oms_rho_ablation.sh manifest.csv results_rho [extra runner args...]

if [ "$#" -lt 2 ]; then
  echo "usage: $0 MANIFEST OUTBASE [extra lazyipdca_cutest args...]" >&2
  exit 2
fi

MANIFEST=$1
OUTBASE=$2
shift 2
EXE=${LAZYIPDCA_CUTEST_BIN:-lazyipdca_cutest}

COMMON=(--manifest "$MANIFEST" --max-iter "${MAX_ITER:-3000}" \
        --time-limit "${TIME_LIMIT:-900}" --eps "${TOL:-1e-6}")
EXTRA=("$@")

run_mode() {
  local tag=$1; shift
  echo "=== OMS rho ablation: $tag ==="
  "$EXE" "${COMMON[@]}" --outdir "${OUTBASE}_${tag}" "$@" "${EXTRA[@]}"
}

run_mode const --regularization-mode const --rho-init "${RHO_INIT:-auto}" \
  --rho-init-rel-margin "${RHO_INIT_REL_MARGIN:-1e-6}" \
  --rho-init-abs-margin "${RHO_INIT_ABS_MARGIN:-1e-8}"

run_mode dyn --regularization-mode dyn --rho-init "${RHO_INIT:-auto}" \
  --rho-init-rel-margin "${RHO_INIT_REL_MARGIN:-1e-6}" \
  --rho-init-abs-margin "${RHO_INIT_ABS_MARGIN:-1e-8}" \
  --rho-trial-every "${RHO_TRIAL_EVERY:-5}" \
  --rho-rel-margin "${RHO_REL_MARGIN:-0.10}" \
  --rho-curvature-tol "${RHO_CURVATURE_TOL:-1e-8}" \
  --rho-lanczos-residual-safety "${RHO_LANCZOS_RESIDUAL_SAFETY:-1.0}" \
  --rho-dyn-floor-rel-margin "${RHO_DYN_FLOOR_REL_MARGIN:-1e-6}" \
  --rho-dyn-reject-growth "${RHO_DYN_REJECT_GROWTH:-2.0}"

run_mode lazy --regularization-mode lazy

echo "Results: ${OUTBASE}_{const,dyn,lazy}"
