#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=${BIN:-/data/projects/IPDCA_QP/lazyipdca_early_lowrank_takeover_v271_source/build_cutest_early_lowrank_takeover_v271/lazyipdca_cutest}
EXPECTED_SHA=${EXPECTED_SHA:-9a767dfa7294279fa914dd4d363c7b0190e6fd6a89c94ab5f46afd19bfeb13d4}
MANIFEST=${MANIFEST:-/data/projects/IPDCA_QP/lazyipdca/manifest_just_nonconvex_extended.csv}
CASES=${CASES:-$HERE/rho_preflight_cases_v275.csv}
ROOT=${ROOT:-/data/projects/IPDCA_QP/RHO_CONST_DYN_PREFLIGHT_v275_$(date +%Y%m%d_%H%M%S)}
fail(){ echo "[FAIL] $*" >&2; exit 2; }
[ -x "$BIN" ] || fail "missing v2.7.1 binary: $BIN"
[ -f "$MANIFEST" ] || fail "missing manifest: $MANIFEST"
[ "$(sha256sum "$BIN" | awk '{print $1}')" = "$EXPECTED_SHA" ] || fail "v2.7.1 SHA mismatch"
python - "$BIN" <<'PY'
from pathlib import Path
import sys
b=Path(sys.argv[1]).read_bytes()
for t in (b'--regularization-mode',b'const',b'dyn',b'lazy'):
    if t not in b: raise SystemExit("[FAIL] scalar mode runtime token missing: "+t.decode())
print("[PASS] exact v2.7.1 binary and CONST/DYN/LAZY CLI surface verified")
PY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COMMON=(--manifest "$MANIFEST" --init 2 --backend ldl --rho-trial-every 2
  --bound-push 0 --bp-push-kappa 1e-4 --bp-margin-tol 1e-4 --escapes 3 --mehrotra
  --sigma-max 1e14 --model-solved-tol 1e-12
  --dca-stop --dca-stall-tol-rel 1e-10 --dca-stall-iters 5 --dca-monotone-tol-abs 1e-6
  --bp-adaptive --max-iter 3000 --time-limit 36000 --eps 1e-6 --verbose 2 --no-atom-cluster
  --dca-momentum original --no-momentum-complete --dca-acceptance lyapunov)
mkdir -p "$ROOT/results" "$ROOT/logs" "$ROOT/records"
: > "$ROOT/runs.jsonl"
run_one(){
  local arm=$1 tag=$2 mode=$3
  local out="$ROOT/results/$arm/$tag" log="$ROOT/logs/${arm}_${tag}.log" rec="$ROOT/records/${arm}_${tag}.json"
  echo "===== $arm $tag mode=$mode ====="
  python "$HERE/run_one_v275.py" --binary "$BIN" --arm "$arm" --tag "$tag"     --outdir "$out" --log "$log" --record "$rec" --     "${COMMON[@]}" --regularization-mode "$mode" --problem "$tag" --outdir "$out"
  cat "$rec" >> "$ROOT/runs.jsonl"
}
while IFS=, read -r tag; do
  [ "$tag" = "tag" ] && continue
  [ -n "$tag" ] || continue
  run_one CONST "$tag" const
  run_one DYN "$tag" dyn
done < "$CASES"
python "$HERE/analyze_rho_preflight_v275.py" "$ROOT"
cp "$ROOT/RHO_PREFLIGHT_PASS" "$HERE/RHO_PREFLIGHT_PASS"
echo "[PASS] CONST/DYN rho preflight complete: $ROOT"
echo "[PASS] campaign gate armed: $HERE/RHO_PREFLIGHT_PASS"
