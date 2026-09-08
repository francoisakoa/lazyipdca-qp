#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python extract_scalar_const_dyn_v275_tables.py "
        "/data/projects/IPDCA_QP/CUTEST_37_scalar_CONST_DYN_v275_..."
    )

ROOT = Path(sys.argv[1])

TAGS = [
    "QPNBOEI2_n143",
    "QPNBOEI1_n384",
    "QPNSTAIR_n385",

    "NCVXBQP1_n1000",
    "NCVXBQP2_n1000",
    "NCVXBQP3_n1000",

    "NCVXQP1_n1000",
    "NCVXQP2_n1000",
    "NCVXQP3_n1000",
    "NCVXQP4_n1000",
    "NCVXQP5_n1000",
    "NCVXQP6_n1000",
    "NCVXQP7_n1000",
    "NCVXQP8_n1000",
    "NCVXQP9_n1000",

    "BLOWEYA_n2002",
    "BLOWEYB_n2002",
    "BLOWEYC_n2002",

    "STNQP1_n8193",
    "STNQP2_n8193",

    "NCVXBQP1_n10000",
    "NCVXBQP2_n10000",
    "NCVXBQP3_n10000",

    "NCVXQP1_n10000",
    "NCVXQP2_n10000",
    "NCVXQP3_n10000",
    "NCVXQP4_n10000",
    "NCVXQP5_n10000",
    "NCVXQP6_n10000",
    "NCVXQP7_n10000",
    "NCVXQP8_n10000",
    "NCVXQP9_n10000",

    "QPNBAND_n10000",

    "BLOCKQP1_n10010",
    "BLOCKQP2_n10010",
    "BLOCKQP3_n10010",

    "QPNBAND_n50000",
]

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def flatten(x, out=None):
    """Flatten nested JSON while preserving the first occurrence of a key."""
    if out is None:
        out = {}

    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict):
                flatten(v, out)
            elif isinstance(v, list):
                # scalar arrays are not needed for this summary
                continue
            elif k not in out:
                out[k] = v

    return out


def pick(d, *names, default=""):
    """Return first existing non-null field."""
    for name in names:
        if name in d and d[name] is not None:
            return d[name]
    return default


def fnum(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def inum(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return ""


def fmt_float(x, digits=6, sci=False):
    try:
        x = float(x)
    except Exception:
        return ""

    if not math.isfinite(x):
        return ""

    if sci:
        return f"{x:.6e}"

    return f"{x:.{digits}f}"


def get_constraint_violation(d):
    # Prefer a directly reported global feasibility/constraint violation.
    direct = pick(
        d,
        "constraint_violation",
        "constraint_violation_inf",
        "constr_violation",
        "constr_viol",
        "primal_violation",
        "primal_inf",
        "feas",
        "feasibility",
        "res_feas_abs",
        default=""
    )

    if direct != "":
        return direct

    # Otherwise reconstruct it from equality and inequality residuals.
    values = []

    for k in (
        "res_eq_abs",
        "eq_abs",
        "eq_violation",
        "equality_violation",
        "res_ineq_abs",
        "ineq_abs",
        "ineq_violation",
        "inequality_violation",
        "bound_violation",
        "box_violation",
    ):
        v = fnum(pick(d, k, default=""))
        if math.isfinite(v):
            values.append(v)

    return max(values) if values else ""


def find_json(resdir, tag):
    cand = [p for p in resdir.rglob("*.json") if tag in str(p)]

    if not cand:
        return None

    # Prefer raw/full result JSON over auxiliary JSON.
    cand.sort(
        key=lambda p: (
            "_raw.json" not in p.name,
            "result" not in p.name.lower(),
            len(str(p)),
        )
    )

    return cand[0]


# ----------------------------------------------------------------------
# Counter aliases
#
# We deliberately support several generations of LazyIPDCA field names.
# ----------------------------------------------------------------------

ALIASES = {
    "objective": (
        "obj",
        "objective",
        "objective_value",
        "f",
        "fval",
    ),

    "iterations": (
        "iters",
        "iterations",
        "iter",
        "n_iter",
    ),

    "time": (
        "time_s",
        "wall_time_s",
        "solve_time_s",
        "solve_time",
        "elapsed_s",
        "elapsed",
        "wall_time",
    ),

    "stationarity": (
        "res_stat_abs",
        "stationarity",
        "stationarity_inf",
        "stat",
        "stat_inf",
        "kkt_stationarity",
    ),

    "complementarity": (
        "res_comp_abs",
        "complementarity",
        "complementarity_inf",
        "comp",
        "comp_inf",
    ),

    # Number of active/generated low-rank corrections / atoms.
    "atoms": (
        "r",
        "rank",
        "rank_hk",
        "hk_rank",
        "n_atoms",
        "num_atoms",
        "atoms",
        "n_rank_one_corrections",
        "rank_one_corrections",
        "n_active_atoms",
    ),

    "lanczos": (
        "n_lanczos",
        "num_lanczos",
        "lanczos_calls",
        "n_lanczos_calls",
        "n_lanczos_runs",
    ),

    "negcurv": (
        "n_negcurv",
        "n_negative_curvature",
        "negative_curvature_calls",
        "n_negative_curvature_events",
        "negcurv",
    ),

    "factorizations": (
        "n_factorizations",
        "factorizations",
        "n_kkt_factorizations",
        "n_kktfac",
        "kktfac",
        "kkt_factorizations",
    ),

    "qv": (
        "n_qv",
        "qv",
        "n_q_matvec",
        "n_q_matvecs",
        "q_matvecs",
    ),

    "rho_initial": (
        "rho_initial",
        "rho_init",
        "initial_rho",
        "rho0",
    ),

    "rho_final": (
        "rho_final",
        "final_rho",
        "rho",
    ),

    "rho_updates": (
        "n_rho_model_updates",
        "n_rho_updates",
        "rho_updates",
    ),

    "rho_increases": (
        "n_rho_model_increases",
        "n_rho_increases",
        "rho_increases",
    ),

    "rho_decreases": (
        "n_rho_model_decreases",
        "n_rho_decreases",
        "rho_decreases",
    ),
}


def get_alias(d, group):
    return pick(d, *ALIASES[group], default="")


# ----------------------------------------------------------------------
# Extract one mode
# ----------------------------------------------------------------------

def extract_mode(mode):
    resdir = ROOT / mode / "results"
    rows = []

    for tag in TAGS:
        jf = find_json(resdir, tag)

        if jf is None:
            rows.append({
                "Problem": tag,
                "n": "",
                "Status": "MISSING_JSON",
                "Obj.": "",
                "Const. viol.": "",
                "Stat.": "",
                "Comp.": "",
                "Iter.": "",
                "Time (s)": "",
                "Rank-one corr.": "",
                "Lanczos": "",
                "Neg. curv.": "",
                "Qv": "",
                "KKT fact.": "",
                "rho_0": "",
                "rho_f": "",
                "rho upd.": "",
                "rho inc.": "",
                "rho dec.": "",
                "JSON": "",
            })
            continue

        try:
            raw = json.loads(jf.read_text())
            d = flatten(raw)
        except Exception:
            rows.append({
                "Problem": tag,
                "n": "",
                "Status": "INVALID_JSON",
                "Obj.": "",
                "Const. viol.": "",
                "Stat.": "",
                "Comp.": "",
                "Iter.": "",
                "Time (s)": "",
                "Rank-one corr.": "",
                "Lanczos": "",
                "Neg. curv.": "",
                "Qv": "",
                "KKT fact.": "",
                "rho_0": "",
                "rho_f": "",
                "rho upd.": "",
                "rho inc.": "",
                "rho dec.": "",
                "JSON": str(jf),
            })
            continue

        status = pick(
            d,
            "status",
            "status_name",
            "solver_status",
            default=""
        )

        rows.append({
            "Problem": tag,
            "n": inum(pick(d, "n", "dimension", default="")),
            "Status": status,

            # Objective should retain enough precision for numerical comparison.
            "Obj.": fmt_float(
                get_alias(d, "objective"),
                sci=True
            ),

            # Residuals: scientific notation is much more useful.
            "Const. viol.": fmt_float(
                get_constraint_violation(d),
                sci=True
            ),
            "Stat.": fmt_float(
                get_alias(d, "stationarity"),
                sci=True
            ),
            "Comp.": fmt_float(
                get_alias(d, "complementarity"),
                sci=True
            ),

            "Iter.": inum(get_alias(d, "iterations")),
            "Time (s)": fmt_float(get_alias(d, "time"), digits=6),

            "Rank-one corr.": inum(get_alias(d, "atoms")),
            "Lanczos": inum(get_alias(d, "lanczos")),
            "Neg. curv.": inum(get_alias(d, "negcurv")),
            "Qv": inum(get_alias(d, "qv")),
            "KKT fact.": inum(get_alias(d, "factorizations")),

            "rho_0": fmt_float(
                get_alias(d, "rho_initial"),
                sci=True
            ),
            "rho_f": fmt_float(
                get_alias(d, "rho_final"),
                sci=True
            ),
            "rho upd.": inum(get_alias(d, "rho_updates")),
            "rho inc.": inum(get_alias(d, "rho_increases")),
            "rho dec.": inum(get_alias(d, "rho_decreases")),

            "JSON": str(jf),
        })

    return rows


FIELDS = [
    "Problem",
    "n",
    "Status",
    "Obj.",
    "Const. viol.",
    "Stat.",
    "Comp.",
    "Iter.",
    "Time (s)",
    "Rank-one corr.",
    "Lanczos",
    "Neg. curv.",
    "Qv",
    "KKT fact.",
    "rho_0",
    "rho_f",
    "rho upd.",
    "rho inc.",
    "rho dec.",
    "JSON",
]


def write_csv(mode, rows):
    out = ROOT / f"table_{mode}.csv"

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    return out


def print_table(mode, rows):
    # More compact terminal version.
    cols = [
        ("Problem", 23),
        ("Status", 16),
        ("Obj.", 15),
        ("Const. viol.", 13),
        ("Iter.", 7),
        ("Time (s)", 11),
        ("Rank-one corr.", 8),
        ("Lanczos", 8),
        ("KKT fact.", 9),
        ("rho_f", 13),
    ]

    print()
    print("=" * 132)
    print(f"{mode} — scalar regularization")
    print("=" * 132)

    header = " ".join(
        f"{name:<{width}}" for name, width in cols
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        vals = []
        for name, width in cols:
            v = str(r.get(name, ""))
            if len(v) > width:
                v = v[:width]
            vals.append(f"{v:<{width}}")
        print(" ".join(vals))


all_rows = {}

for mode in ("CONST", "DYN"):
    rows = extract_mode(mode)
    all_rows[mode] = rows
    out = write_csv(mode, rows)
    print_table(mode, rows)
    print()
    print(f"[written] {out}")


# ----------------------------------------------------------------------
# Diagnostics specifically useful for checking that CONST/DYN are scalar.
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("SCALAR-MODE CONTRACT CHECK")
print("=" * 80)

for mode in ("CONST", "DYN"):
    rows = all_rows[mode]

    missing = sum(r["Status"] == "MISSING_JSON" for r in rows)

    atom_nonzero = []
    for r in rows:
        try:
            if int(r["Rank-one corr."]) != 0:
                atom_nonzero.append(
                    (r["Problem"], int(r["Rank-one corr."]))
                )
        except Exception:
            pass

    lanczos_nonzero = []
    for r in rows:
        try:
            if int(r["Lanczos"]) != 0:
                lanczos_nonzero.append(
                    (r["Problem"], int(r["Lanczos"]))
                )
        except Exception:
            pass

    print(
        f"{mode}: rows={len(rows)}, missing={missing}, "
        f"nonzero_rank_one_corrections={len(atom_nonzero)}, "
        f"nonzero_Lanczos={len(lanczos_nonzero)}"
    )

    if atom_nonzero:
        print("  WARNING rank-one corrections:", atom_nonzero)

    if lanczos_nonzero:
        print("  Lanczos nonzero:", lanczos_nonzero)

print()
print("Generated:")
print(f"  {ROOT / 'table_CONST.csv'}")
print(f"  {ROOT / 'table_DYN.csv'}")
