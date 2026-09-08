#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=${BIN:-/data/projects/IPDCA_QP/lazyipdca_early_lowrank_takeover_v271_source/build_cutest_early_lowrank_takeover_v271/lazyipdca_cutest}
EXPECTED_SHA=${EXPECTED_SHA:-9a767dfa7294279fa914dd4d363c7b0190e6fd6a89c94ab5f46afd19bfeb13d4}
MANIFEST=${MANIFEST:-/data/projects/IPDCA_QP/lazyipdca/manifest_just_nonconvex_extended.csv}
PAIRS=${PAIRS:-$HERE/cutest_37_article_pairs.csv}
ROOT=${ROOT:-/data/projects/IPDCA_QP/SCALAR_VS_LOWRANK_37_v275_$(date +%Y%m%d_%H%M%S)}
fail(){ echo "[FAIL] $*" >&2; exit 2; }
[ -f "$HERE/RHO_PREFLIGHT_PASS" ] || fail "rho preflight has not passed; run ./run_rho_const_dyn_preflight_v275.sh first"
[ -x "$BIN" ] || fail "missing v2.7.1 binary"
[ "$(sha256sum "$BIN" | awk '{print $1}')" = "$EXPECTED_SHA" ] || fail "v2.7.1 SHA mismatch"
[ -f "$MANIFEST" ] || fail "missing manifest"
[ -f "$PAIRS" ] || fail "missing 37-pair list"
echo "[PASS] rho preflight gate present and exact v2.7.1 binary verified"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COMMON=(--manifest "$MANIFEST" --init 2 --backend ldl --rho-trial-every 2
  --bound-push 0 --bp-push-kappa 1e-4 --bp-margin-tol 1e-4 --escapes 3 --mehrotra
  --sigma-max 1e14 --model-solved-tol 1e-12
  --dca-stop --dca-stall-tol-rel 1e-10 --dca-stall-iters 5 --dca-monotone-tol-abs 1e-6
  --bp-adaptive --max-iter 3000 --time-limit 36000 --eps 1e-6 --verbose 1 --no-atom-cluster
  --dca-momentum original --no-momentum-complete --dca-acceptance lyapunov)
LOWRANK_EXTRA=(--adaptive-atoms --max-seed-atoms 0 --max-atoms 500 --seed-coverage 0.25 --seed-stall-tol 0 --seed-patience 999
  --no-adaptive-seed-return --adaptive-marginal-return --adaptive-return-warmup 4 --adaptive-return-window 3
  --adaptive-return-confirmations 2 --adaptive-return-alpha-lift 0.20 --adaptive-return-alpha-effective 0.20
  --adaptive-return-lift-min 1e-5 --adaptive-return-lift-max 1e-3 --adaptive-return-effective-min 1e-5 --adaptive-return-effective-max 2.5e-3
  --adaptive-return-cooldown-factor 2.0 --adaptive-return-cooldown-max-factor 8.0 --adaptive-return-strong-factor 4.0
  --adaptive-return-seed-alpha-coverage 0.20 --adaptive-return-seed-alpha-lift 0.20
  --adaptive-return-seed-coverage-min 1e-5 --adaptive-return-seed-coverage-max 2.5e-3
  --adaptive-return-seed-lift-min 1e-5 --adaptive-return-seed-lift-max 1e-3
  --no-adaptive-return-seed-absolute --adaptive-return-seed-plateau --adaptive-return-seed-plateau-horizon 8
  --adaptive-return-seed-plateau-min-commits 12 --adaptive-return-seed-plateau-confirmations 2
  --adaptive-return-seed-plateau-coverage 2.5e-3 --adaptive-return-seed-plateau-lift 1e-3
  --atom-events --atom-event-every 50 --atom-update-every 20 --atom-every-iters 10 --atoms-per-update 1
  --atom-oracle-min-directional-ratio 0.10)
mkdir -p "$ROOT/results" "$ROOT/logs" "$ROOT/records"
: > "$ROOT/runs.jsonl"
run_one(){
  local arm=$1 tag=$2 mode=$3
  local out="$ROOT/results/$arm/$tag" log="$ROOT/logs/${arm}_${tag}.log" rec="$ROOT/records/${arm}_${tag}.json"
  local -a A
  if [ "$arm" = LOWRANK ]; then A=("${COMMON[@]}" --regularization-mode "$mode" "${LOWRANK_EXTRA[@]}")
  else A=("${COMMON[@]}" --regularization-mode "$mode"); fi
  echo "===== $arm $tag mode=$mode ====="
  python "$HERE/run_one_v275.py" --binary "$BIN" --arm "$arm" --tag "$tag" --outdir "$out" --log "$log" --record "$rec" --     "${A[@]}" --problem "$tag" --outdir "$out"
  cat "$rec" >> "$ROOT/runs.jsonl"
}
count=0
while IFS=, read -r tag; do
  [ "$tag" = "tag" ] && continue
  [ -n "$tag" ] || continue
  count=$((count+1))
  run_one CONST "$tag" const
  run_one DYN "$tag" dyn
  run_one LOWRANK "$tag" lazy
done < "$PAIRS"
[ "$count" -eq 37 ] || fail "pair count=$count, expected 37"
python "$HERE/analyze_scalar_vs_lowrank_37_v275.py" "$ROOT"
echo "[PASS] v2.7.5 scalar-vs-low-rank campaign complete: $ROOT"
