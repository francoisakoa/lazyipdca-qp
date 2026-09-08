#!/usr/bin/env python3
import importlib
import os
import sys

print("===== PREFLIGHT CUTEST37 / QPALM / OSQP =====")
print("python          =", sys.executable)
cache = os.environ.get("PYCUTEST_CACHE", "/data/opt/cutest/cache")
os.environ["PYCUTEST_CACHE"] = cache
os.makedirs(cache, exist_ok=True)
print("PYCUTEST_CACHE  =", cache)
print("MASTSIF         =", os.environ.get("MASTSIF", "<unset>"))
print()

ok = True
for mod in ["numpy", "scipy", "pycutest", "qpalm", "osqp"]:
    try:
        m = importlib.import_module(mod)
        print(f"[PASS] {mod:<10} version={getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        ok = False
        print(f"[MISS] {mod:<10} {type(e).__name__}: {e}")

try:
    import qpalm
    s = qpalm.Settings()
    required = [
        "max_iter", "inner_max_iter", "eps_abs", "eps_rel",
        "eps_abs_in", "eps_rel_in", "eps_prim_inf", "eps_dual_inf",
        "time_limit", "nonconvex",
    ]
    missing = [name for name in required if not hasattr(s, name)]
    if missing:
        ok = False
        print("[MISS] QPALM settings:", ", ".join(missing))
    else:
        print("[PASS] QPALM settings include nonconvex + intermediate tolerances")
        print(f"[INFO] QPALM native inner: eps_abs_in={s.eps_abs_in} eps_rel_in={s.eps_rel_in} inner_max_iter={s.inner_max_iter}")
except Exception as e:
    ok = False
    print(f"[MISS] QPALM settings check: {type(e).__name__}: {e}")

try:
    import osqp
    alg = getattr(osqp, "default_algebra", None)
    if callable(alg):
        print("[INFO] OSQP default algebra =", alg())
except Exception:
    pass

raise SystemExit(0 if ok else 2)
