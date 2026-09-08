#!/usr/bin/env python3
# diagnose_bp_signature.py — v0.16.0
#
# Diagnostic de la strategie adaptive bound push sur une instance en echec :
# execute le solveur OFF puis ON (--bp-adaptive), capture les evenements [BP]
# et rend un verdict : la signature "slack pathologique" est-elle presente,
# quels etages se declenchent, et le statut change-t-il ?
#
# Usage :
#   # depuis une archive npz (cles : Q, q, [A, b, C, cl, cu, l, u, c0])
#   python diagnose_bp_signature.py --npz probleme.npz
#   # depuis CUTEst (pycutest requis, SIF decode)
#   python diagnose_bp_signature.py --cutest NCVXQP7 --sifparams N=1000
#   # options solveur passees telles quelles (ex. point de depart respecte) :
#   python diagnose_bp_signature.py --npz p.npz --bound-push 0
#
# Discipline : mono-thread strict (OMP/BLAS=1) pour reproduire les runs de
# campagne ; les evenements sont dans le stdout du solveur (verbose>=1).
import argparse
import contextlib
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

BP_RE = re.compile(
    r"\[BP\] event=(\d+) n_pushed=(\d+) kappa=([0-9.e+-]+) "
    r"min_slack_before=([0-9.e+-]+|inf|-inf|nan) cause=(\w+)")


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    import scipy.sparse as sp
    def get(k):
        return d[k] if k in d.files else None
    def csc(k):
        v = get(k)
        if v is None:
            return None
        return sp.csc_matrix(v.item() if v.dtype == object else v)
    return dict(Q=csc("Q"), q=get("q"), A=csc("A"), b=get("b"),
                C=csc("C"), cl=get("cl"), cu=get("cu"),
                l=get("l"), u=get("u"),
                c0=float(get("c0")) if get("c0") is not None else 0.0)


def load_cutest(name, sifparams):
    import pycutest
    import scipy.sparse as sp
    params = dict(kv.split("=") for kv in sifparams) if sifparams else None
    if params:
        params = {k: int(v) for k, v in params.items()}
    prob = pycutest.import_problem(name, sifParams=params)
    n = prob.n
    x0 = np.zeros(n)
    _, g = prob.obj(x0, gradient=True)
    H = prob.sphess(x0) if hasattr(prob, "sphess") else sp.csc_matrix(prob.hess(x0))
    Q = sp.csc_matrix(H)
    out = dict(Q=Q, q=g, A=None, b=None, C=None, cl=None, cu=None,
               l=prob.bl, u=prob.bu, c0=float(prob.obj(x0)))
    if prob.m > 0:
        J = sp.csc_matrix(prob.scons(x0)[1] if hasattr(prob, "scons")
                          else prob.cons(x0, gradient=True)[1])
        c0v = prob.cons(x0)
        eq = prob.is_eq_cons
        ineq = ~eq
        if eq.any():
            out["A"] = J[eq]
            out["b"] = (prob.cl[eq] - c0v[eq])
        if ineq.any():
            out["C"] = J[ineq]
            out["cl"] = prob.cl[ineq] - c0v[ineq]
            out["cu"] = prob.cu[ineq] - c0v[ineq]
    return out


@contextlib.contextmanager
def capture_fd_stdout():
    """Capture le stdout au niveau DESCRIPTEUR (les printf C++ contournent
    sys.stdout ; redirect_stdout ne les voit pas)."""
    import tempfile
    fd = sys.stdout.fileno()
    saved = os.dup(fd)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        sys.stdout.flush()
        os.dup2(tmp.fileno(), fd)
        holder = {}
        try:
            yield holder
        finally:
            sys.stdout.flush()
            try:                       # fflush(NULL) : vide les buffers stdio C
                import ctypes
                ctypes.CDLL(None).fflush(None)
            except Exception:
                pass
            os.dup2(saved, fd)
            os.close(saved)
            tmp.seek(0)
            holder["text"] = tmp.read().decode(errors="replace")


def run(P, bp_on, overrides):
    import lazyipdca as lz
    st = lz.Settings()
    st.verbose = 1
    st.bp_adaptive = bp_on
    for k, v in overrides.items():
        setattr(st, k, v)
    with capture_fd_stdout() as cap:
        r = lz.solve(P["Q"], P["q"], P["A"], P["b"], P["C"], P["cl"], P["cu"],
                     P["l"], P["u"], c0=P["c0"], settings=st)
    log = cap["text"]
    info = r["info"] if isinstance(r, dict) else r.info
    events = [m.groups() for m in BP_RE.finditer(log)]
    return info, events, log


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz")
    src.add_argument("--cutest")
    ap.add_argument("--sifparams", nargs="*", default=None)
    ap.add_argument("--bound-push", type=float, default=None)
    ap.add_argument("--bound-frac", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--full-log", action="store_true",
                    help="imprime les stdout complets des deux runs")
    args = ap.parse_args()

    P = load_npz(args.npz) if args.npz else load_cutest(args.cutest, args.sifparams)
    ov = {}
    if args.bound_push is not None:
        ov["bound_push"] = args.bound_push
    if args.bound_frac is not None:
        ov["bound_frac"] = args.bound_frac
    if args.max_iter is not None:
        ov["max_iter"] = args.max_iter

    i0, e0, log0 = run(P, False, ov)
    i1, e1, log1 = run(P, True, ov)

    def fmt(i):
        return (f"status={i.status} it={i.iters} obj={i.obj:+.8e} "
                f"stat_rel={i.res_stat_rel:.2e} feas_rel={i.res_feas_rel:.2e} "
                f"bp_ev={i.n_bp_events} bp_push={i.n_bp_pushed}")

    print("=" * 78)
    print("DIAGNOSTIC ADAPTIVE BOUND PUSH")
    print("-" * 78)
    print("OFF :", fmt(i0))
    print("ON  :", fmt(i1))
    causes = [ev[4] for ev in e1]
    print("-" * 78)
    if not e1 and i1.n_bp_pushed == 0:
        print("VERDICT : AUCUNE signature slack — l'echec (s'il y en a un) ne vient")
        print("  pas des bords. Si un --bound-push manuel aidait quand meme, il")
        print("  agissait par un autre canal (duaux, conditionnement) : m'envoyer")
        print("  le log verbose 2 du run en echec pour affiner la strategie.")
    else:
        print(f"VERDICT : signature detectee — {len(e1)} evenement(s), causes = "
              f"{sorted(set(causes)) if causes else ['interiority/post_anchor']}")
        s0, s1 = int(i0.status), int(i1.status)
        ok = lambda s: s in (0, 1, 7)
        if ok(s1) and not ok(s0):
            print("  => SAUVETAGE de bout en bout : ON converge, OFF echouait.")
        elif ok(s1) and ok(s0):
            print("  => les deux convergent ; comparer iterations/objectif ci-dessus.")
        else:
            print("  => le push seul ne suffit pas ici : la signature est presente")
            print("     mais un autre verrou domine (rho, degenerescence duale...).")
    print("=" * 78)
    if args.full_log:
        print("\n----- LOG OFF -----\n", log0)
        print("\n----- LOG ON ------\n", log1)


if __name__ == "__main__":
    sys.exit(main())
