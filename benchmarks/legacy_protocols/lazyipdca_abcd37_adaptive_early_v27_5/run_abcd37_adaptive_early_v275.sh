#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=${BIN:-/data/projects/IPDCA_QP/lazyipdca_adaptive_early_v275_source/build_cutest_adaptive_early_v275/lazyipdca_cutest}
MANIFEST=${MANIFEST:-/data/projects/IPDCA_QP/lazyipdca/manifest_just_nonconvex_extended.csv}
PAIRS=${PAIRS:-$HERE/cutest_37_article_pairs.csv}
ROOT=${ROOT:-/data/projects/IPDCA_QP/CUTEST_37_ABCD_ADAPTIVE_EARLY_v275_$(date +%Y%m%d_%H%M%S)}
MAX_ITER=${MAX_ITER:-3000}
TIME_LIMIT=${TIME_LIMIT:-36000}
fail(){ echo "[FAIL] $*" >&2; exit 2; }

[ -x "$BIN" ] || fail "missing v2.7.5 binary: $BIN"
[ -f "$MANIFEST" ] || fail "missing manifest: $MANIFEST"
[ -f "$PAIRS" ] || fail "missing pair file: $PAIRS"

python - "$BIN" <<'PY'
from pathlib import Path
import sys
b=Path(sys.argv[1]).read_bytes()
for t in (b'adaptive EARLY TAKEOVER',b'--legacy-atoms',b'--adaptive-atoms',
          b'--max-seed-atoms',b'--adaptive-marginal-return',
          b'--no-momentum-complete',b'--dca-acceptance'):
    if t not in b:
        raise SystemExit('[FAIL] required runtime/CLI token absent: '+t.decode())
print('[PASS] v2.7.5 executable CLI contract')
PY

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

COMMON_PRE=(--manifest "$MANIFEST" --regularization-mode lazy --init 2 --backend ldl
  --rho-trial-every 2 --max-atoms 500 --seed-coverage 0.25
  --seed-stall-tol 0 --seed-patience 999
  --bound-push 0 --bp-push-kappa 1e-4 --bp-margin-tol 1e-4
  --escapes 3 --mehrotra)
COMMON_POST=(--sigma-max 1e14 --model-solved-tol 1e-12
  --dca-stop --dca-stall-tol-rel 1e-10 --dca-stall-iters 5
  --dca-monotone-tol-abs 1e-6 --bp-adaptive
  --max-iter "$MAX_ITER" --time-limit "$TIME_LIMIT" --eps 1e-6
  --verbose 1 --no-atom-cluster)
NO_MOM=(--dca-momentum original --no-momentum-complete --dca-acceptance lyapunov)
EVENTS=(--atom-events --atom-event-every 50 --atom-update-every 20 --atom-every-iters 10 --atoms-per-update 1)
C_RETURN=(--no-adaptive-seed-return --adaptive-marginal-return
  --adaptive-return-warmup 4 --adaptive-return-window 3 --adaptive-return-confirmations 2
  --adaptive-return-alpha-lift 0.20 --adaptive-return-alpha-effective 0.20
  --adaptive-return-lift-min 1e-5 --adaptive-return-lift-max 1e-3
  --adaptive-return-effective-min 1e-5 --adaptive-return-effective-max 2.5e-3
  --adaptive-return-cooldown-factor 2.0 --adaptive-return-cooldown-max-factor 8.0
  --adaptive-return-strong-factor 4.0
  --adaptive-return-seed-alpha-coverage 0.20 --adaptive-return-seed-alpha-lift 0.20
  --adaptive-return-seed-coverage-min 1e-5 --adaptive-return-seed-coverage-max 2.5e-3
  --adaptive-return-seed-lift-min 1e-5 --adaptive-return-seed-lift-max 1e-3
  --no-adaptive-return-seed-absolute --adaptive-return-seed-plateau
  --adaptive-return-seed-plateau-horizon 8 --adaptive-return-seed-plateau-min-commits 12
  --adaptive-return-seed-plateau-confirmations 2 --adaptive-return-seed-plateau-coverage 2.5e-3
  --adaptive-return-seed-plateau-lift 1e-3)

A_ARGS=("${COMMON_PRE[@]}" --legacy-atoms "${COMMON_POST[@]}" "${NO_MOM[@]}" "${EVENTS[@]}" --max-seed-atoms 0)
B_ARGS=("${COMMON_PRE[@]}" --legacy-atoms "${COMMON_POST[@]}" "${NO_MOM[@]}" "${EVENTS[@]}" --max-seed-atoms 20)
C_ARGS=("${COMMON_PRE[@]}" --adaptive-atoms "${COMMON_POST[@]}" "${NO_MOM[@]}" "${EVENTS[@]}" "${C_RETURN[@]}" --max-seed-atoms 0 --atom-oracle-min-directional-ratio 0.10)
D_ARGS=("${COMMON_PRE[@]}" --adaptive-atoms "${COMMON_POST[@]}" "${NO_MOM[@]}" "${EVENTS[@]}" "${C_RETURN[@]}" --max-seed-atoms 20 --atom-oracle-min-directional-ratio 0.10)

echo '[PASS] arm definitions'
echo 'A = coverage-target / no cap / EARLY disabled by adaptive-only gate'
echo 'B = A + cap20'
echo 'C = adaptive selection + EARLY / no cap'
echo 'D = C + cap20 exactly'
echo 'Primary evaluation = feasibility + objective + wall time (not KKT certification)'

mkdir -p "$ROOT/results" "$ROOT/logs" "$ROOT/records"
: > "$ROOT/runs.jsonl"
run_one(){
  local arm=$1 tag=$2
  local out="$ROOT/results/$arm/$tag" log="$ROOT/logs/${arm}_${tag}.log" rec="$ROOT/records/${arm}_${tag}.json"
  local -a ARGS
  case "$arm" in
    A) ARGS=("${A_ARGS[@]}");;
    B) ARGS=("${B_ARGS[@]}");;
    C) ARGS=("${C_ARGS[@]}");;
    D) ARGS=("${D_ARGS[@]}");;
    *) fail "unknown arm $arm";;
  esac
  echo "===== $arm $tag ====="
  python "$HERE/run_one_abcd_v275.py" --binary "$BIN" --arm "$arm" --tag "$tag" \
    --outdir "$out" --log "$log" --record "$rec" -- \
    "${ARGS[@]}" --problem "$tag" --outdir "$out"
  cat "$rec" >> "$ROOT/runs.jsonl"
}

count=0
while IFS=, read -r tag; do
  [ "$tag" = "tag" ] && continue
  [ -n "$tag" ] || continue
  count=$((count+1))
  run_one A "$tag"
  run_one B "$tag"
  run_one C "$tag"
  run_one D "$tag"
done < "$PAIRS"
[ "$count" -eq 37 ] || fail "pair count=$count, expected 37"
python "$HERE/analyze_abcd37_v275.py" "$ROOT"
echo "[PASS] A/B/C/D 37-pair campaign complete: $ROOT"
