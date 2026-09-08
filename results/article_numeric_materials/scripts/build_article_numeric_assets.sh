#!/usr/bin/env bash
set -u
set -o pipefail

BASE=${BASE:-/data/projects/IPDCA_QP}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${OUT:-"$BASE/ARTICLE_READY_NUMERIC"}

mkdir -p "$OUT"

python "$HERE/scripts/validate_authoritative_inputs.py"

cp "$BASE/ARTICLE_NUMERIC_FINAL_C381_MATCHED/three_solver_37_matched.csv" "$OUT/" 2>/dev/null || true
cp "$BASE/ARTICLE_NUMERIC_FINAL_C381_MATCHED/performance_profile_ratios_3solver.csv" "$OUT/" 2>/dev/null || true
cp "$BASE/ARTICLE_NUMERIC_FINAL_C381_MATCHED/performance_profile_curve_3solver.csv" "$OUT/" 2>/dev/null || true
cp "$BASE/ARTICLE_NUMERIC_FINAL_C381_MATCHED/objective_lazy_vs_ipopt.csv" "$OUT/" 2>/dev/null || true
cp "$BASE/ARTICLE_NUMERIC_FINAL_C381_MATCHED/objective_lazy_vs_qpalm.csv" "$OUT/" 2>/dev/null || true

python "$HERE/scripts/make_three_solver_supp_table.py" \
  --input "$BASE/ARTICLE_NUMERIC_FINAL_C381_MATCHED/three_solver_37_matched.csv" \
  --out "$OUT/table_three_solver_37.tex"

python "$HERE/scripts/extract_lcp10.py" \
  --base "$BASE" \
  --out "$OUT/lcp10_discovered.csv"

if python - <<'PY'
try:
    import matplotlib
    print("yes")
except Exception:
    raise SystemExit(1)
PY
then
  python "$HERE/figures/make_performance_profile.py" \
    --input "$BASE/ARTICLE_NUMERIC_FINAL_C381_MATCHED/performance_profile_ratios_3solver.csv" \
    --out "$OUT/performance_profile_3solver.pdf"
else
  echo "[WARN] matplotlib unavailable; performance profile PDF not generated."
fi

cp "$HERE/latex/tables_main.tex" "$OUT/"
cp "$HERE/latex/tables_supplement.tex" "$OUT/"
cp "$HERE/csv/"*.csv "$OUT/"

echo
echo "ARTICLE-READY NUMERIC OUTPUT:"
find "$OUT" -maxdepth 1 -type f -printf '%f\n' | sort
