#!/usr/bin/env python3
"""Fallback CUTEst runner via pycutest — same JSON schema as the C++ runner.

Usage:
    python tools/run_pycutest.py --manifest manifest_full.csv --outdir results \
        [--problem NAME] [--eps 1e-6] [--max-iter 500] [--time-limit 900]

Reads columns 'name' (or first column) and optional 'sif_params'
("N=100000,M=25000" or "N=100000 M=25000").  One output directory per problem
(<outdir>/NAME/NAME_raw.json): safe under GNU-parallel invocation.

Reminder for fair IPOPT comparison (mono-thread discipline):
    env -i HOME=$HOME PATH=$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 python tools/run_pycutest.py ...
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
import numpy as np
import scipy.sparse as sp

import lazyipdca

CUTE_INF = 1e19


def parse_manifest(path):
    """Yields dict(name, tag, params, status, lambda_min_est).

    Supports both the simple schema (name[,sif_params]) and the extended one
    (name,target_n,...,params_json,...,lambda_min_est,...,status,...):
    params_json is a JSON object like {"N": 100000}; entries sharing a name
    are disambiguated by target_n in the tag (NAME_n{target_n})."""
    entries = []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return entries
    header = [c.strip().lower() for c in rows[0]]
    cols = {h: i for i, h in enumerate(header)}
    has_header = any(h in ("name", "problem", "prob") for h in header)
    name_col = next((cols[h] for h in ("name", "problem", "prob") if h in cols), 0)
    start = 1 if has_header else 0

    def cell(row, key):
        i = cols.get(key)
        return row[i].strip() if has_header and i is not None and len(row) > i else ""

    for row in rows[start:]:
        if not row or not row[name_col].strip() or row[name_col].startswith("#"):
            continue
        name = row[name_col].strip()
        params = {}
        pj = cell(row, "params_json")
        if pj:
            try:
                params = {k: int(v) for k, v in json.loads(pj).items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                params = {}
        else:
            sp_ = cell(row, "sif_params") or cell(row, "params") or cell(row, "param")
            for tok in re.split(r"[;, ]+", sp_):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    params[k.strip()] = int(float(v))
        tag = name
        tn = cell(row, "target_n")
        if tn:
            tag = f"{name}_n{tn}"
        lme = cell(row, "lambda_min_est")
        entries.append(dict(name=name, tag=tag, params=params,
                            status=cell(row, "status"),
                            lambda_min_est=float(lme) if lme else float("nan")))
    return entries


def extract_qp(name, sifParams):
    import pycutest
    prob = pycutest.import_problem(name, sifParams=sifParams or None)
    n, mc = prob.n, prob.m
    x0 = np.zeros(n)
    f0, g = prob.obj(x0, gradient=True)
    # Q: Hessian of the Lagrangian at (0, 0).  Constrained problems need
    # ihess/hess with multipliers; the sparse Lagrangian Hessian with v=0
    # equals Q for linear constraints.  (Lesson: use the Lagrangian Hessian,
    # not the objective-only path, on constrained problems.)
    if mc > 0:
        H = prob.sphess(x0, v=np.zeros(mc))
    else:
        H = prob.sphess(x0)
    Q = sp.csc_matrix(H)
    l = np.where(prob.bl <= -CUTE_INF, -np.inf, prob.bl)
    u = np.where(prob.bu >= CUTE_INF, np.inf, prob.bu)
    A = b = C = cl = cu = None
    if mc > 0:
        c0, J = prob.scons(x0, gradient=True)
        J = sp.csc_matrix(J)
        is_eq = np.asarray(prob.is_eq_cons, dtype=bool)
        clo = np.where(prob.cl <= -CUTE_INF, -np.inf, prob.cl)
        cup = np.where(prob.cu >= CUTE_INF, np.inf, prob.cu)
        if is_eq.any():
            A = J[is_eq, :]
            b = clo[is_eq] - c0[is_eq]
        if (~is_eq).any():
            C = J[~is_eq, :]
            cl = np.where(np.isfinite(clo[~is_eq]), clo[~is_eq] - c0[~is_eq], -np.inf)
            cu = np.where(np.isfinite(cup[~is_eq]), cup[~is_eq] - c0[~is_eq], np.inf)
    return dict(Q=Q, q=g, c0=f0, A=A, b=b, C=C, cl=cl, cu=cu, l=l, u=u)


STATUS_STR = ["solved", "solved_soc", "max_iter", "time_limit",
              "primal_infeasible_like", "numerical_error", "unbounded"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--problem", default=None)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--time-limit", type=float, default=1e10)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--only-accepted", action="store_true",
                    help="skip manifest rows whose status != 'accepted'")
    ap.add_argument("--use-manifest-lambda", action="store_true",
                    help="seed the rho cap with the manifest lambda_min_est")
    ap.add_argument("--convdesc", action="store_true",
                    help="enable the v19av convex-descent acceptance relaxation")
    args = ap.parse_args()

    st = lazyipdca.Settings()
    st.eps_stat = st.eps_feas = st.eps_comp = args.eps
    st.max_iter = args.max_iter
    st.time_limit = args.time_limit
    st.verbose = args.verbose
    st.accept_convex_descent = args.convdesc

    fails = 0
    for ent in parse_manifest(args.manifest):
        name, tag, params = ent["name"], ent["tag"], ent["params"]
        if args.problem and name != args.problem and tag != args.problem:
            continue
        if args.only_accepted and ent["status"] != "accepted":
            continue
        if args.use_manifest_lambda and np.isfinite(ent["lambda_min_est"]):
            st.lambda_min_hint = ent["lambda_min_est"]
        else:
            st.lambda_min_hint = float("nan")
        odir = os.path.join(args.outdir, tag)
        os.makedirs(odir, exist_ok=True)
        try:
            data = extract_qp(name, params)
        except Exception as exc:  # decode failure must not kill the batch
            json.dump({"solver": "lazyipdca", "name": name, "status": -1,
                       "status_str": "decode_error", "note": str(exc)},
                      open(os.path.join(odir, f"{tag}_raw.json"), "w"), indent=1)
            print(f"[runner] {name}: DECODE ERROR: {exc}", file=sys.stderr)
            fails += 1
            continue
        t0 = time.perf_counter()
        r = lazyipdca.solve(settings=st, name=tag, **data)
        i = r["info"]
        out = {
            "solver": "lazyipdca", "name": tag, "sif_name": name,
            "n": int(data["q"].size),
            "m_eq": 0 if data["A"] is None else int(data["A"].shape[0]),
            "m_ineq": 0 if data["C"] is None else int(data["C"].shape[0]),
            "status": int(i.status.value), "status_str": STATUS_STR[int(i.status.value)],
            "iters": int(i.iters), "obj": float(i.obj),
            "kkt_stat_abs": float(i.res_stat_abs), "kkt_stat_rel": float(i.res_stat_rel),
            "cv": float(max(i.res_eq_abs, i.res_ineq_abs)),
            "kkt_feas_rel": float(i.res_feas_rel), "kkt_comp_rel": float(i.res_comp_rel),
            "rho_final": float(i.rho_final),
            "lambda_min_reduced": float(i.lambda_min_reduced) if np.isfinite(i.lambda_min_reduced) else 0.0,
            "soc_certified": bool(i.soc_certified),
            "n_factorizations": int(i.n_factorizations),
            "n_inertia_corrections": int(i.n_inertia_corrections),
            "n_lanczos_calls": int(i.n_lanczos_calls),
            "time": float(i.solve_time), "wall_time": time.perf_counter() - t0,
        }
        json.dump(out, open(os.path.join(odir, f"{tag}_raw.json"), "w"), indent=1)
        ok = out["status_str"] in ("solved", "solved_soc")
        fails += 0 if ok else 1
        print(f"[runner] {tag}: {out['status_str']} it={out['iters']} "
              f"obj={out['obj']:+.8e} cv={out['cv']:.2e} stat={out['kkt_stat_rel']:.2e} "
              f"rho={out['rho_final']:.2e} t={out['time']:.2f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
