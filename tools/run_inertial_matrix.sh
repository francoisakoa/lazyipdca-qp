#!/usr/bin/env bash
set -euo pipefail

# Run all 4 x 3 inertial/acceptance combinations with identical remaining
# arguments.  The first three positional arguments are intentionally simple;
# every argument after them is forwarded verbatim to lazyipdca_cutest.
#
# Usage:
#   tools/run_inertial_matrix.sh ./build/lazyipdca_cutest manifest.csv results \
#       --problem NCVXQP3 --backend ldl --max-iter 3000 --eps 1e-6

if [ "$#" -lt 3 ]; then
  echo "usage: $0 RUNNER MANIFEST OUTROOT [extra runner options...]" >&2
  exit 2
fi

runner=$1
manifest=$2
outroot=$3
shift 3

momentum_modes=(original inertiel_euclidien inertiel_global inertiel_atomique)
acceptance_modes=(monotone window lyapunov)

for momentum in "${momentum_modes[@]}"; do
  for acceptance in "${acceptance_modes[@]}"; do
    arm="${momentum}__${acceptance}"
    outdir="${outroot%/}/${arm}"
    printf '[inertial-matrix] %-22s %-8s -> %s\n' "$momentum" "$acceptance" "$outdir"
    "$runner" --manifest "$manifest" --outdir "$outdir" \
      --dca-momentum "$momentum" --dca-acceptance "$acceptance" "$@"
  done
done
