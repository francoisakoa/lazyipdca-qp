# Numerical material for the LazyIPDCA article

This bundle contains **numerical-only** material prepared from the validated campaigns discussed with the reviewers.

## Authoritative CUTEst37 reference

Use configuration C from:

`/data/projects/IPDCA_QP/CUTEST_37_ABCD_v275_20260824_123948/tables_37_abcd/table_C_37.csv`

Total wall time: **381.251494576 s**.

Do **not** use the later 293.487 s C5 reference for the article.

## Main article numbers

- CUTEst37 total wall time:
  - LazyIPDCA C: 381.251 s
  - Ipopt: 3428.100 s
  - QPALM: 6412.156 s
- Aggregate time ratios:
  - Ipopt / LazyIPDCA = 8.99
  - QPALM / LazyIPDCA = 16.82
- Common KKT test at 1e-6:
  - LazyIPDCA: 27/37
  - Ipopt: 28/37
  - QPALM: 35/37
- Objective partitions:
  - LazyIPDCA / Ipopt / ties = 21 / 3 / 13
  - LazyIPDCA / QPALM / ties = 20 / 3 / 14
- Median seconds per iteration:
  - LazyIPDCA: 8.719e-3 s
  - Ipopt: 2.301e-2 s
  - QPALM: 9.270e-4 s
  - LazyIPDCA / QPALM = 9.41

## Performance profile

The exact Dolan--Moré profile should be generated on the TRX from:

`/data/projects/IPDCA_QP/ARTICLE_NUMERIC_FINAL_C381_MATCHED/performance_profile_ratios_3solver.csv`

Use `figures/make_performance_profile.py`.

## QPALM KKT failures on CUTEst37

Both failures are feasible and complementary but miss the common stationarity threshold:
- NCVXBQP3_n10000: stat = 1.809e-6
- BLOCKQP3_n10010: stat = 1.458e-4

## QPALM large-scale panel

Classification: 8 KKT, 3 feasible/non-KKT, 6 infeasible.

The six infeasible cases are:
NCVXQP1_n100000, NCVXQP3_n100000, NCVXQP5_n100000,
NCVXQP6_n100000, NCVXQP9_n100000, BLOWEYA_n100002.

## Sensitivity and timing

- 65-run local sensitivity study: all 65 runs feasible.
- Lanczos cap 30/60/120: no change on 5/5.
- Nonbinding max-atoms 100/500/1000: no change on 5/5.
- Lanczos residual safety 0.5/1/2: no change on 5/5.
- Composite score rewards and penalties scaled by ±25%: no change on 5/5.
- rho_trial_every is the only axis changing trajectories.
- Binding rank cap max_atoms=4 changes trajectories on the instances where it binds.
- Ten-run timing CV: 0.29%--1.24%.

## LCP72

QPALM certificate-matched: 71/72.
Only failure: LCP8, n=5,000,000.

LazyIPDCA: 72/72 certified (aggregate result already used in v85).

### Remaining numerical gap

The LCP10 "unusually fast n=2e6" anomaly still requires a clean extraction of the authoritative LazyIPDCA LCP10 campaign. Use `scripts/extract_lcp10.py`. Do not invent a structural explanation until that extraction is complete.
