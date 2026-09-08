#!/usr/bin/env bash
set -uo pipefail

BASE=/data/projects/IPDCA_QP
SRC="$BASE/lazyipdca_adaptive_early_v275_source"
BIN="$SRC/build_cutest_adaptive_early_v275/lazyipdca_cutest"
MANIFEST="$BASE/lazyipdca/manifest_just_nonconvex_extended.csv"

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="$BASE/CUTEST_37_scalar_CONST_DYN_v275_corrected_${STAMP}"

MAX_ITER="${MAX_ITER:-3000}"
TIME_LIMIT="${TIME_LIMIT:-36000}"
EPS="${EPS:-1e-6}"
RESUME="${RESUME:-1}"

source /home/fakoa/anaconda3/etc/profile.d/conda.sh
conda activate lazydca_atom_oracle

# Do NOT inherit the local XL MASTSIF.
export MASTSIF=/data/opt/cutest/mastsif

export LC_ALL=C
export LANG=C

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export PYCUTEST_CACHE="$ROOT/pycutest_cache"

mkdir -p \
  "$ROOT/CONST/results" "$ROOT/CONST/logs" "$ROOT/CONST/done" \
  "$ROOT/DYN/results"   "$ROOT/DYN/logs"   "$ROOT/DYN/done" \
  "$ROOT/decode" "$PYCUTEST_CACHE"

printf '%s\n' "$ROOT" > "$BASE/last_scalar_const_dyn_v275_root.txt"

TAGS=(
  QPNBOEI2_n143
  QPNBOEI1_n384
  QPNSTAIR_n385

  NCVXBQP1_n1000
  NCVXBQP2_n1000
  NCVXBQP3_n1000

  NCVXQP1_n1000
  NCVXQP2_n1000
  NCVXQP3_n1000
  NCVXQP4_n1000
  NCVXQP5_n1000
  NCVXQP6_n1000
  NCVXQP7_n1000
  NCVXQP8_n1000
  NCVXQP9_n1000

  BLOWEYA_n2002
  BLOWEYB_n2002
  BLOWEYC_n2002

  STNQP1_n8193
  STNQP2_n8193

  NCVXBQP1_n10000
  NCVXBQP2_n10000
  NCVXBQP3_n10000

  NCVXQP1_n10000
  NCVXQP2_n10000
  NCVXQP3_n10000
  NCVXQP4_n10000
  NCVXQP5_n10000
  NCVXQP6_n10000
  NCVXQP7_n10000
  NCVXQP8_n10000
  NCVXQP9_n10000

  QPNBAND_n10000

  BLOCKQP1_n10010
  BLOCKQP2_n10010
  BLOCKQP3_n10010

  QPNBAND_n50000
)

COMMON=(
  --manifest "$MANIFEST"
  --decode-dir "$ROOT/decode"

  --init 2
  --backend ldl

  --max-iter "$MAX_ITER"
  --time-limit "$TIME_LIMIT"
  --eps "$EPS"

  # Same spectral initialisation for CONST and DYN.
  --rho-init auto
  --rho-init-rel-margin 3.0
  --rho-init-abs-margin 1e-8

  --rho-rel-margin 0.10
  --rho-curvature-tol 1e-8
  --rho-dyn-floor-rel-margin 1e-6

  --rho-trial-every 2
  --rho-decay 0.5

  # CORRECTED historical/reference cadence:
  --reject-streak-rho-bump 50

  --rho-dyn-reject-growth 2.0

  --mehrotra

  --dca-stop
  --dca-stall-tol-rel 1e-10
  --dca-stall-iters 5
  --dca-monotone-tol-abs 1e-6

  --bound-push 0
  --bp-push-kappa 1e-4
  --bp-margin-tol 1e-4
  --bp-adaptive

  --escapes 3
  --sigma-max 1e14
  --model-solved-tol 1e-12

  --dca-momentum original
  --dca-acceptance monotone
  --no-momentum-complete

  --verbose 1
)

echo "===================================================================="
echo "SCALAR CONST/DYN v275 — CORRECTED 37-INSTANCE CAMPAIGN"
echo "===================================================================="
echo "ROOT     = $ROOT"
echo "BIN      = $BIN"
echo "MANIFEST = $MANIFEST"
echo "MAX_ITER = $MAX_ITER"
echo "TIME     = $TIME_LIMIT"
echo "EPS      = $EPS"
echo
echo "CRITICAL: reject-streak-rho-bump = 50"
echo

sha256sum "$BIN" | tee "$ROOT/binary.sha256"

{
  echo "campaign=scalar_CONST_DYN_v275_corrected"
  echo "binary=$BIN"
  echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "manifest=$MANIFEST"
  echo "instances=37"
  echo "runs=74"
  echo "max_iter=$MAX_ITER"
  echo "time_limit=$TIME_LIMIT"
  echo "eps=$EPS"
  echo "backend=ldl"
  echo "threads=1"
  echo "rho_init=auto"
  echo "rho_init_rel_margin=3.0"
  echo "rho_rel_margin=0.10"
  echo "rho_trial_every=2"
  echo "rho_decay=0.5"
  echo "reject_streak_rho_bump=50"
  echo "rho_dyn_reject_growth=2.0"
} > "$ROOT/campaign.env"

for MODE in CONST DYN; do

  if [[ "$MODE" == "CONST" ]]; then
    REGMODE=const
  else
    REGMODE=dyn
  fi

  echo
  echo "####################################################################"
  echo "### MODE $MODE"
  echo "####################################################################"

  for TAG in "${TAGS[@]}"; do

    OUT="$ROOT/$MODE/results"
    LOG="$ROOT/$MODE/logs/${TAG}.log"
    DONE="$ROOT/$MODE/done/${TAG}.ok"
    CMD="$ROOT/$MODE/logs/${TAG}.command.txt"

    if [[ "$RESUME" == "1" && -f "$DONE" ]]; then
      echo "[skip] $MODE $TAG"
      continue
    fi

    echo
    echo "===== $MODE $TAG ====="

    RUN=(
      "$BIN"
      "${COMMON[@]}"
      --regularization-mode "$REGMODE"
      --problem "$TAG"
      --outdir "$OUT"
    )

    printf '%q ' "${RUN[@]}" > "$CMD"
    printf '\n' >> "$CMD"

    T0=$(date +%s.%N)

    set +e
    "${RUN[@]}" > "$LOG" 2>&1
    RC=$?
    set -e

    T1=$(date +%s.%N)

    WALL=$(python - "$T0" "$T1" <<'PY'
import sys
print(f"{float(sys.argv[2])-float(sys.argv[1]):.6f}")
PY
)

    JSON=$(find "$OUT" -type f -name '*_raw.json' \
      -path "*${TAG}*" 2>/dev/null | head -1)

    if [[ -n "$JSON" ]]; then
      {
        echo "rc=$RC"
        echo "wall_external_s=$WALL"
        echo "json=$JSON"
      } > "$DONE"

      printf "[run] %-5s %-25s rc=%2d wall=%10ss JSON=yes\n" \
        "$MODE" "$TAG" "$RC" "$WALL"
    else
      printf "[WARN] %-5s %-25s rc=%2d wall=%10ss JSON=NO\n" \
        "$MODE" "$TAG" "$RC" "$WALL"
    fi
  done
done

echo
echo "===================================================================="
echo "CAMPAIGN STATUS"
echo "===================================================================="

for MODE in CONST DYN; do
  NJSON=$(find "$ROOT/$MODE/results" -type f -name '*_raw.json' | wc -l)
  NDONE=$(find "$ROOT/$MODE/done" -type f -name '*.ok' | wc -l)

  echo "$MODE : JSON=$NJSON/37 done=$NDONE/37"
done

echo
echo "ROOT=$ROOT"
echo "[INFO] Campagne terminée/reprenable. Terminal conservé."
