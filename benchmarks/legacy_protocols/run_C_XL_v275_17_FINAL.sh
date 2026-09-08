#!/usr/bin/env bash
set -uo pipefail

BASE=/data/projects/IPDCA_QP
ROOT="$BASE/CUTEST_C_XL_v275_20260825_141918"

SRC="$BASE/lazyipdca_adaptive_early_v275_source"
BIN="$SRC/build_cutest_adaptive_early_v275/lazyipdca_cutest"

MANIFEST="$ROOT/manifest_C_XL_v275_FINAL.csv"
MASTSIF_XL="$ROOT/MASTSIF_XL_LOCAL"

OUTDIR="$ROOT/results_C17_FINAL"
LOGDIR="$ROOT/logs_C17_FINAL"
DONEDIR="$ROOT/done_C17_FINAL"
DECODEDIR="$ROOT/decode_C17_FINAL"

EXPECTED_SHA256="1b4744dc400d3948120804c79f47532bc192b018462b6bb953c1b11f82d93f43"

MAX_ITER="${MAX_ITER:-3000}"
TIME_LIMIT="${TIME_LIMIT:-36000}"
EPS="${EPS:-1e-6}"
RESUME="${RESUME:-1}"

source /home/fakoa/anaconda3/etc/profile.d/conda.sh
conda activate lazydca_atom_oracle

export MASTSIF="$MASTSIF_XL"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$OUTDIR" "$LOGDIR" "$DONEDIR" "$DECODEDIR"

RC=0

echo
echo "===================================================================="
echo "C-XL v275 — 17 INSTANCES"
echo "===================================================================="
echo "BIN      = $BIN"
echo "MANIFEST = $MANIFEST"
echo "MASTSIF  = $MASTSIF"
echo "OUTDIR   = $OUTDIR"
echo

# ------------------------------------------------------------------
# 1. Binary / manifest / MASTSIF checks
# ------------------------------------------------------------------

if [ ! -x "$BIN" ]; then
    echo "[FAIL] binaire v275 absent/non exécutable:"
    echo "       $BIN"
    RC=1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "[FAIL] manifeste absent: $MANIFEST"
    RC=1
fi

if [ ! -d "$MASTSIF" ]; then
    echo "[FAIL] MASTSIF XL absent: $MASTSIF"
    RC=1
fi

if [ "$RC" -eq 0 ]; then
    ACTUAL_SHA256=$(sha256sum "$BIN" | awk '{print $1}')
    echo "SHA256   = $ACTUAL_SHA256"

    if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
        echo "[FAIL] SHA256 différent de la référence v275 attendue."
        echo "expected = $EXPECTED_SHA256"
        echo "actual   = $ACTUAL_SHA256"
        RC=1
    else
        echo "[PASS] binaire v275 byte-identifié."
    fi
fi

# ------------------------------------------------------------------
# 2. Verify exactly 17 validated instances
# ------------------------------------------------------------------

if [ "$RC" -eq 0 ]; then
    NINST=$(python - "$MANIFEST" "$MASTSIF" <<'PY'
import csv,sys
from pathlib import Path

manifest=Path(sys.argv[1])
mastsif=Path(sys.argv[2])

with manifest.open(newline="") as f:
    rows=list(csv.DictReader(f))

ok=True
for r in rows:
    if not (mastsif/f"{r['name']}.SIF").is_file():
        ok=False
        print("[FAIL] missing SIF:",r["name"])

print(len(rows) if ok else -1)
PY
)

    if [ "$NINST" != "17" ]; then
        echo "[FAIL] panel final != 17 instances: $NINST"
        RC=1
    else
        echo "[PASS] panel XL = 17/17."
    fi
fi

# ------------------------------------------------------------------
# 3. Check critical v275 CLI contract BEFORE running anything
# ------------------------------------------------------------------

if [ "$RC" -eq 0 ]; then
    STRINGS_FILE="$ROOT/v275_C17_binary_strings.txt"
    strings "$BIN" > "$STRINGS_FILE"

    REQUIRED_FLAGS=(
      "--manifest"
      "--decode-dir"
      "--problem"
      "--outdir"
      "--regularization-mode"
      "--adaptive-atoms"
      "--adaptive-marginal-return"
      "--init"
      "--backend"
      "--rho-init"
      "--rho-init-rel-margin"
      "--rho-init-abs-margin"
      "--rho-rel-margin"
      "--rho-curvature-tol"
      "--rho-dyn-floor-rel-margin"
      "--rho-trial-every"
      "--rho-decay"
      "--rho-dyn-reject-growth"
      "--reject-streak-rho-bump"
      "--max-atoms"
      "--seed-coverage"
      "--seed-stall-tol"
      "--seed-patience"
      "--atom-events"
      "--atom-event-every"
      "--mehrotra"
      "--dca-stop"
      "--dca-stall-tol-rel"
      "--dca-stall-iters"
      "--dca-monotone-tol-abs"
      "--dca-momentum"
      "--dca-acceptance"
      "--no-momentum-complete"
    )

    echo
    echo "===== v275 CLI CONTRACT ====="

    for flag in "${REQUIRED_FLAGS[@]}"; do
        if grep -Fq -- "$flag" "$STRINGS_FILE"; then
            printf "[PASS] %s\n" "$flag"
        else
            printf "[FAIL] %s absent du binaire\n" "$flag"
            RC=1
        fi
    done
fi

# ------------------------------------------------------------------
# 4. Configuration C
# ------------------------------------------------------------------

COMMON=(
  --manifest "$MANIFEST"
  --decode-dir "$DECODEDIR"

  --init 2
  --backend ldl

  --regularization-mode lazy
  --adaptive-atoms
  --adaptive-marginal-return

  --rho-init auto
  --rho-init-rel-margin 1e-6
  --rho-init-abs-margin 1e-8

  --rho-rel-margin 0.10
  --rho-curvature-tol 1e-8
  --rho-dyn-floor-rel-margin 1e-6

  --rho-trial-every 2
  --rho-decay 0.5
  --rho-dyn-reject-growth 2.0

  --reject-streak-rho-bump 50

  --max-atoms 500
  --seed-coverage 0.25
  --seed-stall-tol 0
  --seed-patience 999

  --atom-events
  --atom-event-every 50

  --bound-push 0
  --bp-push-kappa 1e-4
  --bp-margin-tol 1e-4
  --bp-adaptive

  --escapes 3
  --mehrotra
  --sigma-max 1e14
  --model-solved-tol 1e-12

  --dca-stop
  --dca-stall-tol-rel 1e-10
  --dca-stall-iters 5
  --dca-monotone-tol-abs 1e-6

  --dca-momentum original
  --dca-acceptance monotone
  --no-momentum-complete

  --max-iter "$MAX_ITER"
  --time-limit "$TIME_LIMIT"
  --eps "$EPS"
  --verbose 1
)

# ------------------------------------------------------------------
# 5. Save exact campaign provenance
# ------------------------------------------------------------------

if [ "$RC" -eq 0 ]; then
    {
        echo "campaign=C_XL_v275_17_FINAL"
        echo "date=$(date --iso-8601=seconds)"
        echo "binary=$BIN"
        echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
        echo "manifest=$MANIFEST"
        echo "mastsif=$MASTSIF"
        echo "threads=1"
        echo "max_iter=$MAX_ITER"
        echo "time_limit=$TIME_LIMIT"
        echo "eps=$EPS"
        echo "reject_streak_rho_bump=50"
        echo "momentum=original/off"
    } > "$ROOT/C_XL_v275_17_FINAL_provenance.txt"

    printf '%q ' "$BIN" "${COMMON[@]}" \
        > "$ROOT/C_XL_v275_17_FINAL_common_command.txt"
    echo >> "$ROOT/C_XL_v275_17_FINAL_common_command.txt"
fi

# ------------------------------------------------------------------
# 6. Read exact tags from final validated manifest
# ------------------------------------------------------------------

if [ "$RC" -eq 0 ]; then
    mapfile -t TAGS < <(
      python - "$MANIFEST" <<'PY'
import csv,sys
with open(sys.argv[1],newline="") as f:
    for r in csv.DictReader(f):
        print(f"{r['name']}_n{r['target_n']}")
PY
    )

    echo
    echo "===== RUN ORDER ====="
    printf '%s\n' "${TAGS[@]}"
    echo
fi

# ------------------------------------------------------------------
# 7. Run all 17 sequentially; resume-safe
# ------------------------------------------------------------------

if [ "$RC" -eq 0 ]; then

    for TAG in "${TAGS[@]}"; do

        LOG="$LOGDIR/${TAG}.log"
        CMD="$LOGDIR/${TAG}.command.txt"
        DONE="$DONEDIR/${TAG}.done"

        if [ "$RESUME" = "1" ] && [ -f "$DONE" ]; then
            echo "[skip] $TAG"
            continue
        fi

        echo
        echo "===================================================================="
        echo "C  $TAG"
        echo "===================================================================="

        RUN=(
          "$BIN"
          "${COMMON[@]}"
          --problem "$TAG"
          --outdir "$OUTDIR"
        )

        {
            printf 'MASTSIF=%q ' "$MASTSIF"
            printf 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
            printf '%q ' "${RUN[@]}"
            printf '\n'
        } > "$CMD"

        T0=$(date +%s.%N)

        set +e
        "${RUN[@]}" > "$LOG" 2>&1
        RUN_RC=$?
        set -e

        T1=$(date +%s.%N)

        WALL=$(python - "$T0" "$T1" <<'PY'
import sys
print(f"{float(sys.argv[2])-float(sys.argv[1]):.6f}")
PY
)

        JSON="$OUTDIR/$TAG/${TAG}_raw.json"

        if [ ! -f "$JSON" ]; then
            JSON=$(find "$OUTDIR" -type f -name "${TAG}_raw.json" \
                   2>/dev/null | head -1)
        fi

        if [ -n "$JSON" ] && [ -f "$JSON" ]; then
            {
                echo "rc=$RUN_RC"
                echo "wall_external_s=$WALL"
                echo "json=$JSON"
            } > "$DONE"

            python - "$TAG" "$JSON" "$WALL" <<'PY'
import json,sys

tag,jf,wall=sys.argv[1:]
d=json.load(open(jf))

print(
    f"[run] {tag:<24} "
    f"status={str(d.get('status_str',d.get('status','?'))):<18} "
    f"it={str(d.get('iters','?')):>5} "
    f"obj={float(d.get('obj',float('nan'))):+.6e} "
    f"cv={float(d.get('cv',float('nan'))):.3e} "
    f"rank={str(d.get('n_atoms',d.get('rank','?'))):>4} "
    f"L={str(d.get('n_lanczos_calls','?')):>5} "
    f"KKT={str(d.get('n_factorizations','?')):>5} "
    f"wall={float(wall):.3f}s"
)
PY
        else
            echo "[FAIL] $TAG : aucun JSON ; rc=$RUN_RC wall=${WALL}s"
        fi
    done
fi

# ------------------------------------------------------------------
# 8. Final audit/table
# ------------------------------------------------------------------

if [ "$RC" -eq 0 ]; then
    python - "$ROOT" "$MANIFEST" <<'PY'
import csv,json,math,sys
from pathlib import Path

root=Path(sys.argv[1])
manifest=Path(sys.argv[2])
outdir=root/"results_C17_FINAL"
done=root/"done_C17_FINAL"

with manifest.open(newline="") as f:
    specs=list(csv.DictReader(f))

rows=[]

for s in specs:
    name=s["name"]
    n=int(s["target_n"])
    tag=f"{name}_n{n}"

    jf=outdir/tag/f"{tag}_raw.json"

    if not jf.exists():
        hits=list(outdir.rglob(f"{tag}_raw.json"))
        jf=hits[0] if hits else None

    external=""
    df=done/f"{tag}.done"
    if df.exists():
        for line in df.read_text().splitlines():
            if line.startswith("wall_external_s="):
                external=line.split("=",1)[1]

    if jf is None or not jf.exists():
        rows.append({
            "problem":name,"n":n,"m":s["m"],"status":"MISSING",
            "objective":"","constraint_violation":"","iterations":"",
            "time_s":"","wall_external_s":external,
            "rank_one_corrections":"","lanczos_calls":"",
            "kkt_factorizations":"","rho_final":"",
            "feasible_1e-6":"NO",
        })
        continue

    d=json.loads(jf.read_text())

    cv=d.get("cv","")
    try:
        feasible=float(cv) <= 1e-6
    except Exception:
        feasible=False

    rows.append({
        "problem":name,
        "n":d.get("n",n),
        "m":int(d.get("m_eq",0))+int(d.get("m_ineq",0))
            if ("m_eq" in d or "m_ineq" in d) else s["m"],
        "status":d.get("status_str",d.get("status","")),
        "objective":d.get("obj",""),
        "constraint_violation":cv,
        "iterations":d.get("iters",""),
        "time_s":d.get("time",d.get("wall_time","")),
        "wall_external_s":external,
        "rank_one_corrections":d.get(
            "n_atoms",
            d.get("rank",d.get("rank_hk",""))
        ),
        "lanczos_calls":d.get("n_lanczos_calls",""),
        "kkt_factorizations":d.get("n_factorizations",""),
        "rho_final":d.get("rho_final",""),
        "feasible_1e-6":"YES" if feasible else "NO",
    })

fields=list(rows[0].keys())
csvout=root/"C_XL_v275_17_FINAL_results.csv"

with csvout.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print()
print("="*150)
print("C-XL v275 — FINAL TABLE")
print("="*150)
print(
    f"{'Problem':<12}{'n':>9}{'m':>9}{'status':>18}"
    f"{'objective':>19}{'cv':>13}{'iter':>7}"
    f"{'t(s)':>11}{'rank':>7}{'Lanczos':>9}{'KKT':>8}{'feas':>7}"
)
print("-"*150)

for r in rows:
    def fmt(v,form):
        try: return format(float(v),form)
        except: return str(v)

    print(
        f"{r['problem']:<12}"
        f"{str(r['n']):>9}"
        f"{str(r['m']):>9}"
        f"{str(r['status']):>18}"
        f"{fmt(r['objective'],'.6e'):>19}"
        f"{fmt(r['constraint_violation'],'.3e'):>13}"
        f"{str(r['iterations']):>7}"
        f"{fmt(r['time_s'],'.3f'):>11}"
        f"{str(r['rank_one_corrections']):>7}"
        f"{str(r['lanczos_calls']):>9}"
        f"{str(r['kkt_factorizations']):>8}"
        f"{r['feasible_1e-6']:>7}"
    )

present=sum(r["status"]!="MISSING" for r in rows)
feasible=sum(r["feasible_1e-6"]=="YES" for r in rows)

print()
print(f"JSON      = {present}/17")
print(f"feasible  = {feasible}/17   (absolute cv <= 1e-6)")
print(f"CSV       = {csvout}")
PY
fi

echo
echo "===================================================================="
echo "C-XL v275 terminé/reprenable"
echo "ROOT = $ROOT"
echo "===================================================================="
echo "[INFO] Le terminal reste ouvert."
