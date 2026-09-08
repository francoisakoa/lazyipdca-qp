# v4 decode correction

This revision preserves the v3 native-dual KKT logic and fixes CUTEst decoding for the fixed-size problems QPNBOEI2, QPNBOEI1, and QPNSTAIR. Their historical `target_n` labels in the 37-instance manifest are not passed to SIFDECODE as `N`.

The runner supports `--resume`; therefore an existing v3 campaign directory can be completed in place after upgrading the kit.

# CUTEst 37 — QPALM + OSQP benchmark kit v3

This revision contains QPALM and OSQP only.

## v3 corrections

1. **Native dual multipliers are now primary for the KKT check.**
   - QPALM: `solver.solution.y`
   - OSQP: `res.y`
   - bounded active-set least squares is retained only as a diagnostic fallback.
2. `stat_abs_recovered` and `stat_rel_recovered` are reported so the old
   active-set reconstruction can be compared with the native-dual residual.
3. QPALM's intermediate parameters are **not forced** by default.
   `eps_abs_in`, `eps_rel_in`, and `inner_max_iter` remain at QPALM native
   defaults unless the optional CLI flags are explicitly supplied.
4. Final benchmark tolerances remain explicit: `eps_abs=TOL`, `eps_rel=0`,
   `eps_prim_inf=TOL`, `eps_dual_inf=TOL`, `nonconvex=1`.

## Experimental contract

- manifest: `/data/projects/IPDCA_QP/lazyipdca/manifest_just_nonconvex_extended.csv`
- original CUTEst coordinates and original `x0`
- fixed variables retained
- one CPU thread
- wall-time cap: 36,000 s per run
- common KKT tolerance: `1e-6`
- aggregate CSV: `results_37_qpalm_osqp.csv`

The common post-check reports primal feasibility, stationarity,
complementarity and multiplier-sign consistency. `kkt1_ok` is a first-order
KKT test, not a global-optimality certificate.

## Recommended QPALM smoke test

```bash
cd /data/projects/IPDCA_QP/cutest37_qpalm_osqp
conda activate qpsolvers37

rm -rf /data/projects/IPDCA_QP/CUTEST_37_QPALM_SMOKE_V3

python bench37.py \
  --manifest /data/projects/IPDCA_QP/lazyipdca/manifest_just_nonconvex_extended.csv \
  --outdir /data/projects/IPDCA_QP/CUTEST_37_QPALM_SMOKE_V3 \
  --solvers qpalm \
  --only NCVXQP1_n1000 \
  --max-iter 100000 \
  --time-limit 36000 \
  --tol 1e-6
```

Do **not** pass `--qpalm-inner-tol` or `--qpalm-inner-max-iter` for the baseline.
Those switches remain available for explicit sensitivity experiments.

## Full campaign

```bash
MAX_ITER=100000 \
TIME_LIMIT=36000 \
TOL=1e-6 \
bash run_37.sh
```

QPALM only:

```bash
SOLVERS=qpalm bash run_37.sh
```

OSQP only:

```bash
SOLVERS=osqp bash run_37.sh
```

OSQP is not called on clearly indefinite Hessians unless `OSQP_FORCE=1`; such
forced diagnostic rows remain `comparable=False`.


## v5 decode corrections

The v5 runner pins the PyCUTEst cache to `/data/opt/cutest/cache` by default and overrides stale `PYCUTEST_CACHE` values. It also uses exact CUTEst SIF parameters for the eight historically failing tags:

- BLOWEYA/B/C_n2002: `N=1000` (2*N+2 = 2002 variables)
- STNQP1/2_n8193: `P=13` (2**P+1 = 8193 variables)
- BLOCKQP1/2/3_n10010: `N=5000, B=10` (2*N+B = 10010 variables)

These are parameterized CUTEst models; BLOCKQP is not treated as fixed-size.
