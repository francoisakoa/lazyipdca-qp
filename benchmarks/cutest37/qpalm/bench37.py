#!/usr/bin/env python3
"""Benchmark QPALM and OSQP on the 37 LazyIPDCA CUTEst QP instances.

The script reconstructs each QP in original CUTEst coordinates, keeps fixed
variables, starts from CUTEst x0, uses the manifest SIF parameters, and writes
one JSON per run plus an aggregate CSV.

First-order optimality is checked independently of solver termination by a
common KKT post-check:
  - absolute primal feasibility,
  - relative/absolute stationarity,
  - relative/absolute complementarity,
  - KKT1_OK verdict.
The common post-check uses each solver's native dual multipliers when they are
available. Bounded sparse least squares on the near-active set is retained only
as a diagnostic fallback if a solver does not return a usable dual vector.

OSQP is a convex-QP solver. By default an indefinite Hessian is recorded as
unsupported_nonconvex and is NOT sent to OSQP. --osqp-force is diagnostic only
and the resulting row remains comparable=False.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


TAGS37 = [
    "QPNBOEI2_n143", "QPNBOEI1_n384", "QPNSTAIR_n385",
    "NCVXBQP1_n1000", "NCVXBQP2_n1000", "NCVXBQP3_n1000",
    "NCVXQP1_n1000", "NCVXQP2_n1000", "NCVXQP3_n1000",
    "NCVXQP4_n1000", "NCVXQP5_n1000", "NCVXQP6_n1000",
    "NCVXQP7_n1000", "NCVXQP8_n1000", "NCVXQP9_n1000",
    "BLOWEYA_n2002", "BLOWEYB_n2002", "BLOWEYC_n2002",
    "STNQP1_n8193", "STNQP2_n8193",
    "NCVXBQP1_n10000", "NCVXBQP2_n10000", "NCVXBQP3_n10000",
    "NCVXQP1_n10000", "NCVXQP2_n10000", "NCVXQP3_n10000",
    "NCVXQP4_n10000", "NCVXQP5_n10000", "NCVXQP6_n10000",
    "NCVXQP7_n10000", "NCVXQP8_n10000", "NCVXQP9_n10000",
    "QPNBAND_n10000",
    "BLOCKQP1_n10010", "BLOCKQP2_n10010", "BLOCKQP3_n10010",
    "QPNBAND_n50000",
]

INF_CUTEST = 1.0e19

# These CUTEst problems are fixed-size.  The campaign manifest uses target_n
# only to form the historical benchmark tag; N is not a valid SIF parameter
# for these models.  Passing N causes SIFDECODE to fail.
FIXED_SIZE_NO_SIFPARAMS = {
    "QPNBOEI2",
    "QPNBOEI1",
    "QPNSTAIR",
}

# Exact SIF parameters required to reproduce the historical 37-set dimensions.
# These are derived from the CUTEst SIF definitions, not inferred from the
# benchmark tag.  In particular:
#   BLOWEY*: 2*N+2 variables -> N=2000 gives n=4002 (37-set target)
#   STNQP*:  2**P+1 variables -> P=13 gives n=8193
#   BLOCKQP*: 2*N+B variables -> N=5000,B=10 gives n=10010
SIF_PARAM_OVERRIDES = {
    "BLOWEYA_n2002": {"N": 2000},
    "BLOWEYB_n2002": {"N": 2000},
    "BLOWEYC_n2002": {"N": 2000},
    "STNQP1_n8193": {"P": 13},
    "STNQP2_n8193": {"P": 13},
    "BLOCKQP1_n10010": {"N": 5000, "B": 10},
    "BLOCKQP2_n10010": {"N": 5000, "B": 10},
    "BLOCKQP3_n10010": {"N": 5000, "B": 10},
}

# Expected dimensions after applying the exact SIF overrides above.
# These dimensions, not the historical manifest target_n labels, define
# comparability with the LazyIPDCA 37-instance reference campaign.
EXPECTED_OVERRIDE_DIMS = {
    "BLOWEYA_n2002": (4002, 2002),
    "BLOWEYB_n2002": (4002, 2002),
    "BLOWEYC_n2002": (4002, 2002),
    "STNQP1_n8193": (8193, 4095),
    "STNQP2_n8193": (8193, 4095),
    "BLOCKQP1_n10010": (10010, 5001),
    "BLOCKQP2_n10010": (10010, 5001),
    "BLOCKQP3_n10010": (10010, 5001),
}


@dataclass
class ManifestEntry:
    name: str
    tag: str
    sif_params: Dict[str, Any]
    target_n: Optional[int]
    row: Dict[str, str]


@dataclass
class QPData:
    tag: str
    problem: str
    n: int
    m_original: int
    f0: float
    q: np.ndarray
    Q: sp.csc_matrix
    J: sp.csc_matrix
    cl: np.ndarray
    cu: np.ndarray
    bl: np.ndarray
    bu: np.ndarray
    Aall: sp.csc_matrix
    lall: np.ndarray
    uall: np.ndarray
    x0: np.ndarray
    lambda_min_Q: float
    lambda_min_method: str
    q_norm_inf: float
    Q_norm_inf: float


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _pick_col(row: Dict[str, str], names: Sequence[str]) -> str:
    nr = {_norm_key(k): (v or "") for k, v in row.items() if k is not None}
    for name in names:
        if _norm_key(name) in nr:
            return nr[_norm_key(name)].strip()
    return ""


def _coerce_scalar(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return s
        try:
            if re.fullmatch(r"[-+]?\d+", s):
                return int(s)
            return float(s)
        except Exception:
            return s
    return v


def _parse_params(row: Dict[str, str]) -> Dict[str, Any]:
    js = _pick_col(row, ["params_json"])
    if js:
        try:
            obj = json.loads(js)
            if isinstance(obj, dict):
                return {str(k): _coerce_scalar(v) for k, v in obj.items()}
        except Exception as exc:
            raise ValueError(f"invalid params_json={js!r}: {exc}") from exc

    raw = _pick_col(row, ["sif_params", "params", "param"])
    out: Dict[str, Any] = {}
    if raw:
        for tok in re.split(r"[;\s]+", raw.strip()):
            if not tok or "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            out[k.strip()] = _coerce_scalar(v)
    return out


def read_manifest(path: Path) -> Dict[str, ManifestEntry]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")

    out: Dict[str, ManifestEntry] = {}
    for row in rows:
        name = _pick_col(row, ["name", "problem", "prob"])
        if not name:
            # Fallback: first CSV field, matching the existing C++ runner.
            name = (next(iter(row.values())) or "").strip()
        if not name:
            continue

        target_s = _pick_col(row, ["target_n"])
        target_n = None
        if target_s:
            try:
                target_n = int(float(target_s))
            except Exception:
                pass
        tag = f"{name}_n{target_n}" if target_n is not None else name
        out[tag] = ManifestEntry(
            name=name,
            tag=tag,
            sif_params=_parse_params(row),
            target_n=target_n,
            row=row,
        )
    return out


def cutest_to_inf(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    a[a >= INF_CUTEST] = np.inf
    a[a <= -INF_CUTEST] = -np.inf
    return a


def sparse_inf_norm(M: sp.spmatrix) -> float:
    if M.shape[0] == 0 or M.nnz == 0:
        return 0.0
    row_sums = np.asarray(np.abs(M).sum(axis=1)).ravel()
    return float(np.max(row_sums)) if row_sums.size else 0.0


def estimate_lambda_min(Q: sp.csc_matrix, tol: float = 1e-8) -> Tuple[float, str]:
    n = Q.shape[0]
    if n == 0:
        return 0.0, "empty"
    if n <= 300:
        vals = np.linalg.eigvalsh(Q.toarray())
        return float(vals[0]), "dense_eigvalsh"
    try:
        vals = eigsh(Q, k=1, which="SA", return_eigenvectors=False,
                     tol=tol, maxiter=max(1000, min(20000, 10*n)))
        return float(vals[0]), "eigsh_SA"
    except ArpackNoConvergence as exc:
        if exc.eigenvalues is not None and len(exc.eigenvalues):
            return float(np.min(exc.eigenvalues)), "eigsh_partial"
    except Exception:
        pass

    # Safe diagnostic fallback: Gershgorin lower bound. If this is negative it
    # does NOT prove indefiniteness, so mark the method explicitly.
    diag = Q.diagonal()
    radii = np.asarray(np.abs(Q).sum(axis=1)).ravel() - np.abs(diag)
    lb = float(np.min(diag - radii)) if n else 0.0
    return lb, "gershgorin_lower_bound"


def build_qp(entry: ManifestEntry, cache_prefix: str) -> QPData:
    import pycutest

    destination = re.sub(r"[^A-Za-z0-9_]", "_", f"{cache_prefix}_{entry.tag}")
    sif_params = SIF_PARAM_OVERRIDES.get(entry.tag, entry.sif_params or None)
    if entry.tag in SIF_PARAM_OVERRIDES:
        print(
            f"[DECODE-PARAMS] {entry.tag}: manifest={entry.sif_params!r} "
            f"-> using {sif_params!r}",
            flush=True,
        )

    # QPNBOEI1/2 and QPNSTAIR are fixed CUTEst models.  Their manifest
    # target_n values are benchmark labels, not legal SIF parameter values.
    # In particular QPNSTAIR_n385 decodes to the native CUTEst dimension, so
    # target_n must never be used as a post-decode dimension assertion.
    if entry.name in FIXED_SIZE_NO_SIFPARAMS:
        if entry.sif_params:
            print(
                f"[DECODE-FIXED] {entry.tag}: ignoring manifest SIF params "
                f"{entry.sif_params!r}; CUTEst model is fixed-size",
                flush=True,
            )
        prob = pycutest.import_problem(
            entry.name,
            destination=destination + "_fixed",
            sifParams=None,
            quiet=True,
            drop_fixed_variables=False,
        )
    else:
        prob = pycutest.import_problem(
            entry.name,
            destination=destination,
            sifParams=sif_params,
            quiet=True,
            drop_fixed_variables=False,
        )

    n = int(prob.n)
    m = int(prob.m)
    if entry.tag in EXPECTED_OVERRIDE_DIMS:
        expected_n, expected_m = EXPECTED_OVERRIDE_DIMS[entry.tag]
        if n != expected_n or m != expected_m:
            raise RuntimeError(
                f"{entry.tag}: decoded n={n}, m={m}; expected "
                f"n={expected_n}, m={expected_m} with SIF parameters {sif_params!r}"
            )
        print(
            f"[DECODE-CHECK] {entry.tag}: decoded n={n} m={m} "
            f"matches expected dimensions",
            flush=True,
        )
    xz = np.zeros(n, dtype=float)
    f0, g0 = prob.obj(xz, gradient=True)
    q = np.asarray(g0, dtype=float).reshape(-1)

    Q = prob.isphess(xz).tocsc().astype(float)
    Q = ((Q + Q.T) * 0.5).tocsc()
    Q.eliminate_zeros()

    if m:
        linear_flags = np.asarray(prob.is_linear_cons, dtype=bool)
        if linear_flags.size != m or not np.all(linear_flags):
            raise RuntimeError(f"{entry.tag}: benchmark expects linear constraints only")
        c0, J = prob.scons(xz, gradient=True)
        J = J.tocsc().astype(float)
        c0 = np.asarray(c0, dtype=float).reshape(-1)
        cl = cutest_to_inf(np.asarray(prob.cl, dtype=float).reshape(-1) - c0)
        cu = cutest_to_inf(np.asarray(prob.cu, dtype=float).reshape(-1) - c0)
    else:
        J = sp.csc_matrix((0, n), dtype=float)
        cl = np.empty(0, dtype=float)
        cu = np.empty(0, dtype=float)

    bl = cutest_to_inf(np.asarray(prob.bl, dtype=float).reshape(-1))
    bu = cutest_to_inf(np.asarray(prob.bu, dtype=float).reshape(-1))
    x0 = np.asarray(prob.x0, dtype=float).reshape(-1).copy()

    I = sp.eye(n, format="csc")
    Aall = sp.vstack([J, I], format="csc")
    lall = np.concatenate([cl, bl])
    uall = np.concatenate([cu, bu])

    lmin, lmethod = estimate_lambda_min(Q)
    return QPData(
        tag=entry.tag,
        problem=entry.name,
        n=n,
        m_original=m,
        f0=float(f0),
        q=q,
        Q=Q,
        J=J,
        cl=cl,
        cu=cu,
        bl=bl,
        bu=bu,
        Aall=Aall,
        lall=lall,
        uall=uall,
        x0=x0,
        lambda_min_Q=lmin,
        lambda_min_method=lmethod,
        q_norm_inf=float(np.linalg.norm(q, ord=np.inf)) if n else 0.0,
        Q_norm_inf=sparse_inf_norm(Q),
    )


def qp_objective(d: QPData, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(d.f0 + 0.5 * x.dot(d.Q @ x) + d.q.dot(x))


def _max0(a: np.ndarray) -> float:
    return float(np.max(a)) if a.size else 0.0


def _finite_max_abs(*arrays: np.ndarray) -> float:
    best = 0.0
    for arr in arrays:
        a = np.asarray(arr, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size:
            best = max(best, float(np.max(np.abs(a))))
    return best


def primal_feasibility(A: sp.csc_matrix, l: np.ndarray, u: np.ndarray,
                       x: np.ndarray) -> Tuple[float, float, np.ndarray]:
    z = np.asarray(A @ x, dtype=float).reshape(-1)
    low = np.zeros_like(z)
    up = np.zeros_like(z)
    fl = np.isfinite(l)
    fu = np.isfinite(u)
    low[fl] = np.maximum(l[fl] - z[fl], 0.0)
    up[fu] = np.maximum(z[fu] - u[fu], 0.0)
    feas_abs = max(_max0(low), _max0(up))
    scale = 1.0 + max(_finite_max_abs(l, u), _finite_max_abs(z))
    return feas_abs, feas_abs / scale, z


def recover_kkt_multiplier(d: QPData, x: np.ndarray, active_tol: float,
                           lsmr_tol: float, max_iter: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    A, l, u = d.Aall, d.lall, d.uall
    z = np.asarray(A @ x).ravel()
    grad = np.asarray(d.Q @ x + d.q).ravel()
    m = A.shape[0]

    finite_l = np.isfinite(l)
    finite_u = np.isfinite(u)
    eq = finite_l & finite_u & (np.abs(u - l) <= active_tol)
    near_l = finite_l & ((z - l) <= active_tol)
    near_u = finite_u & ((u - z) <= active_tol)
    active = eq | near_l | near_u
    idx = np.flatnonzero(active)

    y = np.zeros(m, dtype=float)
    if idx.size == 0:
        return y, {"method": "active_bvls_lsq", "active": 0, "status": "no_active_constraints"}

    # Stationarity convention: grad + A^T y = 0.
    # For l <= Ax <= u: y<=0 on lower-only, y>=0 on upper-only, free on equality.
    lb = np.full(idx.size, -np.inf)
    ub = np.full(idx.size,  np.inf)
    for j, i in enumerate(idx):
        if eq[i] or (near_l[i] and near_u[i]):
            continue
        if near_l[i] and not near_u[i]:
            ub[j] = 0.0
        elif near_u[i] and not near_l[i]:
            lb[j] = 0.0

    B = A[idx, :].T.tocsr()
    try:
        res = lsq_linear(
            B, -grad, bounds=(lb, ub), method="trf",
            tol=lsmr_tol, lsmr_tol=lsmr_tol,
            max_iter=max_iter, verbose=0,
        )
        y[idx] = np.asarray(res.x, dtype=float)
        status = f"lsq_status_{res.status}"
        optimal_cost = float(res.cost)
    except Exception as exc:
        status = f"lsq_failed:{type(exc).__name__}:{exc}"
        optimal_cost = math.nan

    return y, {
        "method": "active_bvls_lsq",
        "active": int(idx.size),
        "status": status,
        "least_squares_cost": optimal_cost,
    }


def _kkt_metrics_from_y(d: QPData, x: np.ndarray, y: np.ndarray,
                        tol: float, active_tol: float,
                        source: str, status: str,
                        active_count: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate common first-order KKT residuals for a supplied dual vector.

    Sign convention for l <= A x <= u:
      grad f(x) + A^T y = 0,
      y <= 0 on lower-only active constraints,
      y >= 0 on upper-only active constraints.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    feas_abs, feas_rel, z = primal_feasibility(d.Aall, d.lall, d.uall, x)

    grad = np.asarray(d.Q @ x + d.q).ravel()
    aty = np.asarray(d.Aall.T @ y).ravel()
    stat_vec = grad + aty
    stat_abs = float(np.linalg.norm(stat_vec, ord=np.inf)) if stat_vec.size else 0.0
    stat_scale = 1.0 + max(
        d.q_norm_inf,
        float(np.linalg.norm(d.Q @ x, ord=np.inf)) if d.n else 0.0,
        float(np.linalg.norm(aty, ord=np.inf)) if aty.size else 0.0,
    )
    stat_rel = stat_abs / stat_scale

    l, u = d.lall, d.uall
    finite_l = np.isfinite(l)
    finite_u = np.isfinite(u)
    eq = finite_l & finite_u & (np.abs(u - l) <= active_tol)

    slack_l = np.zeros_like(z)
    slack_u = np.zeros_like(z)
    slack_l[finite_l] = np.maximum(z[finite_l] - l[finite_l], 0.0)
    slack_u[finite_u] = np.maximum(u[finite_u] - z[finite_u], 0.0)

    # QPALM/OSQP interval multiplier convention: negative = lower side,
    # positive = upper side. Split the net interval multiplier accordingly.
    y_lo = np.maximum(-y, 0.0)
    y_up = np.maximum(y, 0.0)
    comp_l = y_lo[finite_l] * slack_l[finite_l]
    comp_u = y_up[finite_u] * slack_u[finite_u]
    comp_abs = max(_max0(comp_l), _max0(comp_u))
    yinf = float(np.linalg.norm(y, ord=np.inf)) if y.size else 0.0
    bscale = max(1.0, _finite_max_abs(l, u, z))
    comp_rel = comp_abs / (1.0 + yinf * bscale)

    # Explicit sign check is only needed when a row has one finite side.
    lower_only = finite_l & ~finite_u
    upper_only = finite_u & ~finite_l
    dual_sign_abs = max(
        _max0(np.maximum(y[lower_only], 0.0)),
        _max0(np.maximum(-y[upper_only], 0.0)),
    )

    if active_count is None:
        active_count = int(np.count_nonzero(np.abs(y) > tol))

    kkt_ok = bool(
        feas_abs <= tol and
        stat_rel <= tol and
        comp_rel <= tol and
        dual_sign_abs <= tol
    )
    return {
        "feas_abs": feas_abs,
        "feas_rel": feas_rel,
        "stat_abs": stat_abs,
        "stat_rel": stat_rel,
        "comp_abs": comp_abs,
        "comp_rel": comp_rel,
        "dual_sign_abs": dual_sign_abs,
        "kkt1_ok": kkt_ok,
        "kkt_check": status,
        "kkt_multiplier_source": source,
        "kkt_active": active_count,
    }


def kkt_postcheck(d: QPData, x: Optional[np.ndarray], tol: float,
                  active_tol: float, kkt_lsq_tol: float,
                  kkt_lsq_max_iter: int,
                  y_solver: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if x is None:
        return {
            "feas_abs": math.nan, "feas_rel": math.nan,
            "stat_abs": math.nan, "stat_rel": math.nan,
            "stat_abs_recovered": math.nan, "stat_rel_recovered": math.nan,
            "comp_abs": math.nan, "comp_rel": math.nan,
            "dual_sign_abs": math.nan,
            "kkt1_ok": False, "kkt_check": "no_primal_point",
            "kkt_multiplier_source": "none", "kkt_active": 0,
        }
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != d.n or not np.all(np.isfinite(x)):
        return {
            "feas_abs": math.nan, "feas_rel": math.nan,
            "stat_abs": math.nan, "stat_rel": math.nan,
            "stat_abs_recovered": math.nan, "stat_rel_recovered": math.nan,
            "comp_abs": math.nan, "comp_rel": math.nan,
            "dual_sign_abs": math.nan,
            "kkt1_ok": False, "kkt_check": "invalid_primal_point",
            "kkt_multiplier_source": "none", "kkt_active": 0,
        }

    # Always compute the old active-set recovery as a diagnostic reference.
    y_rec, rec = recover_kkt_multiplier(
        d, x, active_tol, kkt_lsq_tol, kkt_lsq_max_iter
    )
    rec_metrics = _kkt_metrics_from_y(
        d, x, y_rec, tol, active_tol,
        source=rec.get("method", "active_bvls_lsq"),
        status=rec.get("status", ""),
        active_count=rec.get("active", 0),
    )

    # Primary verdict: use the solver's own dual multipliers whenever possible.
    native_ok = False
    if y_solver is not None:
        try:
            y_native = np.asarray(y_solver, dtype=float).reshape(-1)
            native_ok = (
                y_native.size == d.Aall.shape[0]
                and np.all(np.isfinite(y_native))
            )
        except Exception:
            native_ok = False

    if native_ok:
        out = _kkt_metrics_from_y(
            d, x, y_native, tol, active_tol,
            source="solver_native_dual",
            status="native_dual",
        )
    else:
        out = rec_metrics.copy()
        out["kkt_check"] = "fallback_" + str(out.get("kkt_check", ""))

    out["stat_abs_recovered"] = rec_metrics.get("stat_abs", math.nan)
    out["stat_rel_recovered"] = rec_metrics.get("stat_rel", math.nan)
    out["kkt_lsq_cost"] = rec.get("least_squares_cost", math.nan)
    return out


def first_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return default


def solver_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", getattr(mod, "version", "unknown")))
    except Exception:
        return "unavailable"


def run_qpalm(d: QPData, args: argparse.Namespace) -> Dict[str, Any]:
    import qpalm

    data = qpalm.Data(d.n, d.Aall.shape[0])
    data.Q = d.Q
    data.q = d.q
    data.A = d.Aall
    # Use QPALM's large-finite-bound convention rather than IEEE infinities.
    data.bmin = np.where(np.isneginf(d.lall), -1.0e20, np.where(np.isposinf(d.lall), 1.0e20, d.lall))
    data.bmax = np.where(np.isposinf(d.uall),  1.0e20, np.where(np.isneginf(d.uall), -1.0e20, d.uall))

    settings = qpalm.Settings()
    settings.max_iter = int(args.max_iter)
    settings.eps_abs = float(args.tol)
    settings.eps_rel = 0.0
    # Intermediate tolerances are algorithmic parameters, not the benchmark
    # KKT threshold. Preserve QPALM's native defaults unless explicitly set.
    if args.qpalm_inner_tol is not None:
        settings.eps_abs_in = float(args.qpalm_inner_tol)
        settings.eps_rel_in = 0.0
    if args.qpalm_inner_max_iter is not None:
        settings.inner_max_iter = int(args.qpalm_inner_max_iter)
    settings.eps_prim_inf = float(args.tol)
    settings.eps_dual_inf = float(args.tol)
    settings.time_limit = float(args.time_limit)
    settings.nonconvex = 1
    settings.verbose = 0
    try:
        settings.warm_start = 1
    except Exception:
        pass

    solver = qpalm.Solver(data, settings)
    try:
        solver.warm_start(d.x0, np.zeros(d.Aall.shape[0], dtype=float))
    except Exception:
        pass

    t0 = time.perf_counter()
    solver.solve()
    wall = time.perf_counter() - t0

    x = np.asarray(solver.solution.x, dtype=float).reshape(-1).copy()
    y = np.asarray(solver.solution.y, dtype=float).reshape(-1).copy()
    info = solver.info
    status = str(first_attr(info, ["status"], "unknown"))
    iters = first_attr(info, ["iter", "iters", "iterations", "iter_out"], None)
    try:
        iters = int(iters) if iters is not None else None
    except Exception:
        iters = None
    return {
        "x": x,
        "y": y,
        "status": status,
        "status_code": first_attr(info, ["status_val", "status_code"], None),
        "iterations": iters,
        "wall_time_s": float(wall),
        "solver_reported_time_s": first_attr(info, ["run_time", "solve_time", "setup_time"], None),
        "native_prim_res": first_attr(info, ["pri_res", "prim_res", "primal_residual"], None),
        "native_dual_res": first_attr(info, ["dua_res", "dual_res", "dual_residual"], None),
        "comparable": True,
        "note": (
            "QPALM nonconvex=1; native dual used for KKT; "
            + (f"eps_abs_in={args.qpalm_inner_tol:g}; " if args.qpalm_inner_tol is not None else "eps_abs_in=native; ")
            + (f"inner_max_iter={args.qpalm_inner_max_iter}" if args.qpalm_inner_max_iter is not None else "inner_max_iter=native")
        ),
    }


def osqp_is_supported(d: QPData, convex_tol: float) -> Tuple[bool, str]:
    scale = max(1.0, d.Q_norm_inf)
    threshold = convex_tol * scale
    if d.lambda_min_method == "gershgorin_lower_bound":
        # A negative Gershgorin lower bound cannot prove indefiniteness.
        return False, "convexity_not_certified_eigensolver_failed"
    if d.lambda_min_Q < -threshold:
        return False, f"indefinite_Q(lambda_min={d.lambda_min_Q:.6e},threshold={-threshold:.6e})"
    return True, f"PSD_within_tol(lambda_min={d.lambda_min_Q:.6e},threshold={-threshold:.6e})"


def run_osqp(d: QPData, args: argparse.Namespace) -> Dict[str, Any]:
    supported, reason = osqp_is_supported(d, args.osqp_convex_tol)
    if not supported and not args.osqp_force:
        return {
            "x": None,
            "status": "unsupported_nonconvex",
            "status_code": 9,
            "iterations": 0,
            "wall_time_s": 0.0,
            "solver_reported_time_s": 0.0,
            "native_prim_res": math.nan,
            "native_dual_res": math.nan,
            "comparable": False,
            "note": reason,
        }

    import osqp

    solver = osqp.OSQP()
    P = sp.triu(d.Q, format="csc")
    setup_kw = dict(
        P=P, q=d.q, A=d.Aall, l=d.lall, u=d.uall,
        verbose=False,
        max_iter=int(args.max_iter),
        eps_abs=float(args.tol),
        eps_rel=0.0,
        eps_prim_inf=float(args.tol),
        eps_dual_inf=float(args.tol),
        time_limit=float(args.time_limit),
        polishing=True,
        scaled_termination=False,
        warm_starting=True,
    )
    t0 = time.perf_counter()
    solver.setup(**setup_kw)
    try:
        solver.warm_start(x=d.x0)
    except Exception:
        pass
    res = solver.solve(raise_error=False)
    wall = time.perf_counter() - t0
    info = res.info
    x = None if res.x is None else np.asarray(res.x, dtype=float).reshape(-1).copy()
    y = None if res.y is None else np.asarray(res.y, dtype=float).reshape(-1).copy()
    return {
        "x": x,
        "y": y,
        "status": str(info.status),
        "status_code": int(info.status_val),
        "iterations": int(info.iter),
        "wall_time_s": float(wall),
        "solver_reported_time_s": float(info.run_time),
        "native_prim_res": float(info.prim_res),
        "native_dual_res": float(info.dual_res),
        "comparable": bool(supported),
        "note": reason + ("; forced diagnostic run" if not supported else ""),
    }



def json_safe(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
        return None
    if isinstance(v, Path):
        return str(v)
    return v


def clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: json_safe(v) for k, v in rec.items() if k not in {"x", "y"}}


def write_json(path: Path, record: Dict[str, Any], x: Optional[np.ndarray],
               y: Optional[np.ndarray] = None) -> None:
    obj = clean_record(record)
    obj["x"] = None if x is None else np.asarray(x, dtype=float).tolist()
    obj["y"] = None if y is None else np.asarray(y, dtype=float).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = {k: json_safe(r.get(k)) for k in fields}
            w.writerow(rr)
    tmp.replace(path)


def make_base_record(solver: str, d: QPData, version: str) -> Dict[str, Any]:
    return {
        "solver": solver,
        "solver_version": version,
        "tag": d.tag,
        "problem": d.problem,
        "n": d.n,
        "m": d.m_original,
        "nnz_Q": int(d.Q.nnz),
        "nnz_J": int(d.J.nnz),
        "lambda_min_Q": d.lambda_min_Q,
        "lambda_min_method": d.lambda_min_method,
        "objective": math.nan,
        "feas_abs": math.nan,
        "feas_rel": math.nan,
        "stat_abs": math.nan,
        "stat_rel": math.nan,
        "stat_abs_recovered": math.nan,
        "stat_rel_recovered": math.nan,
        "comp_abs": math.nan,
        "comp_rel": math.nan,
        "dual_sign_abs": math.nan,
        "kkt1_ok": False,
        "kkt_check": "not_run",
        "kkt_multiplier_source": "none",
        "kkt_active": 0,
        "status": "not_run",
        "status_code": None,
        "iterations": None,
        "minor_iterations": None,
        "wall_time_s": math.nan,
        "solver_reported_time_s": math.nan,
        "native_prim_res": math.nan,
        "native_dual_res": math.nan,
        "comparable": True,
        "solver_max_iter": None,
        "time_limit_s": None,
        "kkt_tol": None,
        "active_tol": None,
        "qpalm_inner_tol": None,
        "qpalm_inner_max_iter": None,
        "note": "",
    }


CSV_FIELDS = [
    "solver", "solver_version", "tag", "problem", "n", "m",
    "nnz_Q", "nnz_J", "lambda_min_Q", "lambda_min_method",
    "comparable", "status", "status_code", "iterations", "minor_iterations",
    "objective", "wall_time_s", "solver_reported_time_s",
    "feas_abs", "feas_rel", "stat_abs", "stat_rel",
    "stat_abs_recovered", "stat_rel_recovered",
    "comp_abs", "comp_rel", "dual_sign_abs", "kkt1_ok",
    "kkt_check", "kkt_multiplier_source", "kkt_active",
    "native_prim_res", "native_dual_res",
    "solver_max_iter", "time_limit_s", "kkt_tol", "active_tol",
    "qpalm_inner_tol", "qpalm_inner_max_iter", "note",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--solvers", default="qpalm,osqp",
                   help="comma-separated subset: qpalm,osqp")
    p.add_argument("--only", action="append", default=[], help="run only this tag (repeatable)")
    p.add_argument("--max-iter", type=int, default=100000,
                   help="solver iteration safeguard; wall-time remains the cross-solver resource cap")
    p.add_argument("--time-limit", type=float, default=36000.0)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--active-tol", type=float, default=1e-5,
                   help="near-active tolerance for common multiplier recovery")
    p.add_argument("--kkt-lsq-tol", type=float, default=1e-8)
    p.add_argument("--kkt-lsq-max-iter", type=int, default=200)
    p.add_argument("--osqp-convex-tol", type=float, default=1e-10,
                   help="PSD threshold relative to max(1,||Q||inf)")
    p.add_argument("--osqp-force", action="store_true",
                   help="diagnostic only: try OSQP even if Q is indefinite; row remains non-comparable")
    p.add_argument("--qpalm-inner-tol", type=float, default=None,
                   help="optional QPALM intermediate absolute tolerance; omitted = native QPALM default")
    p.add_argument("--qpalm-inner-max-iter", type=int, default=None,
                   help="optional QPALM per-subproblem iteration cap; omitted = native QPALM default")
    p.add_argument("--cache-prefix", default="bench37qps")
    p.add_argument("--pycutest-cache", default="/data/opt/cutest/cache",
                   help="PyCUTEst cache directory; overrides any stale PYCUTEST_CACHE environment value")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-save-x", action="store_true", help="omit x from per-run JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.pycutest_cache).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYCUTEST_CACHE"] = str(cache_dir)

    requested_solvers = [s.strip().lower() for s in args.solvers.split(",") if s.strip()]
    bad = [s for s in requested_solvers if s not in {"qpalm", "osqp"}]
    if bad:
        print(f"[FAIL] unknown solver(s): {bad}")
        return 2

    manifest = read_manifest(args.manifest)
    missing_tags = [t for t in TAGS37 if t not in manifest]
    if missing_tags:
        print("[FAIL] 37-set tags missing from manifest:", *missing_tags, sep="\n  ")
        return 2

    tags = args.only if args.only else TAGS37
    for t in tags:
        if t not in TAGS37:
            print(f"[FAIL] --only tag not in canonical 37 set: {t}")
            return 2

    versions = {
        "qpalm": solver_version("qpalm"),
        "osqp": solver_version("osqp"),
    }

    print("===== CUTEST 37 / QPALM + OSQP =====")
    print("manifest      =", args.manifest)
    print("outdir        =", args.outdir)
    print("pycutest cache=", cache_dir)
    print("solvers       =", " ".join(requested_solvers))
    print("instances     =", len(tags))
    print("max iter      =", args.max_iter)
    print("time limit    =", args.time_limit)
    print("KKT tol       =", args.tol)
    print("active tol    =", args.active_tol)
    print("QPALM in tol =", args.qpalm_inner_tol if args.qpalm_inner_tol is not None else "native")
    print("QPALM in max =", args.qpalm_inner_max_iter if args.qpalm_inner_max_iter is not None else "native")
    print("OSQP force    =", int(args.osqp_force))
    print()

    rows: List[Dict[str, Any]] = []
    aggregate_path = args.outdir / "results_37_qpalm_osqp.csv"

    for itag, tag in enumerate(tags, 1):
        entry = manifest[tag]
        print(f"===== [{itag:02d}/{len(tags):02d}] {tag} =====", flush=True)
        try:
            d = build_qp(entry, args.cache_prefix)
            print(f"[QP] {entry.name}: decoded n={d.n} m={d.m_original} "
                  f"nnzQ={d.Q.nnz} nnzJ={d.J.nnz} "
                  f"lambda_min(Q)={d.lambda_min_Q:.6e} ({d.lambda_min_method})", flush=True)
        except Exception as exc:
            print(f"[FAIL-DECODE] {tag}: {exc}", flush=True)
            for solver_name in requested_solvers:
                rec = {
                    "solver": solver_name, "solver_version": versions.get(solver_name, ""),
                    "tag": tag, "problem": entry.name, "n": None, "m": None,
                    "status": "decode_error", "comparable": False,
                    "note": f"{type(exc).__name__}: {exc}", "kkt1_ok": False,
                }
                rows.append(rec)
            atomic_write_csv(aggregate_path, rows, CSV_FIELDS)
            continue

        for solver_name in requested_solvers:
            inst_dir = args.outdir / solver_name / tag
            json_path = inst_dir / "result.json"
            if args.resume and json_path.exists():
                try:
                    old = json.loads(json_path.read_text())
                    rows.append(old)
                    print(f"[RESUME] {solver_name:<6} {tag}")
                    continue
                except Exception:
                    pass

            rec = make_base_record(solver_name, d, versions.get(solver_name, ""))
            rec.update({
                "solver_max_iter": int(args.max_iter),
                "time_limit_s": float(args.time_limit),
                "kkt_tol": float(args.tol),
                "active_tol": float(args.active_tol),
                "qpalm_inner_tol": (float(args.qpalm_inner_tol) if solver_name == "qpalm" and args.qpalm_inner_tol is not None else None),
                "qpalm_inner_max_iter": (int(args.qpalm_inner_max_iter) if solver_name == "qpalm" and args.qpalm_inner_max_iter is not None else None),
            })
            raw: Dict[str, Any] = {}
            x: Optional[np.ndarray] = None
            y: Optional[np.ndarray] = None
            try:
                if solver_name == "qpalm":
                    raw = run_qpalm(d, args)
                elif solver_name == "osqp":
                    raw = run_osqp(d, args)
                x = raw.pop("x", None)
                y = raw.pop("y", None)
                rec.update(raw)
                if x is not None:
                    rec["objective"] = qp_objective(d, x)
                kkt = kkt_postcheck(
                    d, x, args.tol, args.active_tol,
                    args.kkt_lsq_tol, args.kkt_lsq_max_iter,
                    y_solver=y,
                )
                rec.update(kkt)
            except Exception as exc:
                rec.update({
                    "status": "exception",
                    "comparable": False if solver_name == "osqp" else rec.get("comparable", True),
                    "note": f"{type(exc).__name__}: {exc}",
                    "kkt1_ok": False,
                    "kkt_check": "solver_exception",
                })
                (inst_dir / "exception.txt").parent.mkdir(parents=True, exist_ok=True)
                (inst_dir / "exception.txt").write_text(traceback.format_exc())

            save_x = None if args.no_save_x else x
            write_json(json_path, rec, save_x, None if args.no_save_x else y)
            rows.append(clean_record(rec))
            atomic_write_csv(aggregate_path, rows, CSV_FIELDS)

            obj = rec.get("objective", math.nan)
            feas = rec.get("feas_abs", math.nan)
            stat = rec.get("stat_rel", math.nan)
            stat_rec = rec.get("stat_rel_recovered", math.nan)
            comp = rec.get("comp_rel", math.nan)
            wall = rec.get("wall_time_s", math.nan)
            src = str(rec.get("kkt_multiplier_source", "none"))
            print(
                f"[run] {solver_name:<6} {tag:<24} "
                f"status={str(rec.get('status',''))[:24]:<24} "
                f"obj={obj if obj is not None else math.nan:.9e} "
                f"feas={feas if feas is not None else math.nan:.3e} "
                f"stat_rel={stat if stat is not None else math.nan:.3e} "
                f"stat_rec={stat_rec if stat_rec is not None else math.nan:.3e} "
                f"comp_rel={comp if comp is not None else math.nan:.3e} "
                f"KKT={int(bool(rec.get('kkt1_ok', False)))} "
                f"dual={src} "
                f"wall={wall if wall is not None else math.nan:.6f}s "
                f"cmp={int(bool(rec.get('comparable', False)))}",
                flush=True,
            )
        print()

    # Rebuild aggregate from all per-run JSONs so --resume / partial invocations remain stable.
    all_rows: List[Dict[str, Any]] = []
    for solver_name in ["qpalm", "osqp"]:
        for tag in TAGS37:
            p = args.outdir / solver_name / tag / "result.json"
            if p.exists():
                try:
                    all_rows.append(json.loads(p.read_text()))
                except Exception:
                    pass
    atomic_write_csv(aggregate_path, all_rows, CSV_FIELDS)

    print("===== SUMMARY =====")
    for solver_name in requested_solvers:
        sr = [r for r in all_rows if r.get("solver") == solver_name]
        n_present = len(sr)
        n_comp = sum(bool(r.get("comparable")) for r in sr)
        n_feas = sum((r.get("feas_abs") is not None and isinstance(r.get("feas_abs"), (int,float)) and r.get("feas_abs") <= args.tol) for r in sr)
        n_kkt = sum(bool(r.get("kkt1_ok")) for r in sr)
        total = sum(float(r.get("wall_time_s") or 0.0) for r in sr if r.get("wall_time_s") is not None)
        print(f"{solver_name:<6}: rows={n_present:2d}/37 comparable={n_comp:2d}/37 "
              f"feasible={n_feas:2d}/37 KKT1_OK={n_kkt:2d}/37 total_wall={total:.6f}s")
    print("CSV =", aggregate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
