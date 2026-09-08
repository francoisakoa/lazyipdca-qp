# Reproducibility protocol

## Hardware and common controls

The reported reference runs used an AMD Ryzen Threadripper 7960X workstation with 256 GiB memory, GCC 13.3, and one computational thread per run.

Common CUTEst rules are: original starting point; fixed variables retained; feasibility threshold `1e-6`; objective tie band `1e-4`; LazyIPDCA and Ipopt safeguards of 3000 iterations; QPALM safeguard of 50000 iterations; wall-time limit 36000 s; common KKT1 threshold `1e-6`.

## Software environments

`environment.yml` builds the LazyIPDCA/CUTEst environment. `environment-benchmarks.yml` pins the principal Python comparator stack used for reproducibility, including PyCUTEst 1.8.2, Ipopt 3.14.19/cyipopt 1.7.0, QPALM 1.3.0 and OSQP 1.1.3.

`bash scripts/install_cutest.sh` installs the traditional CUTEst/SIFDecode/MASTSIF layout required by the native CUTEst runner and writes an environment file under `$HOME/opt/cutest/env.sh`.

## CUTEst37

`benchmarks/manifests/cutest37.csv` defines the 37-instance panel. The exact A--D LazyIPDCA protocol is in `benchmarks/cutest37/run_lazyipdca_abcd.sh`; the two scalar regularization arms are in `run_scalar_ablation.sh`.

The Ipopt comparator is reproduced with `benchmarks/ipopt/run_cutest37.sh`. The QPALM/OSQP extraction and KKT post-check are in `benchmarks/cutest37/qpalm/`.

The authoritative configuration-C reference total is `381.251494576` seconds. Numerical reference tables are under `results/reference/`.

## XL17

`benchmarks/manifests/xl17.csv` lists the 17 rows. Nine rows use official/predefined sizes. Eight rows are parametric extensions of STNQP1--2, BLOWEYA--C and BLOCKQP1--3 made from local copies of their official SIF files by changing only the active numerical size parameter. CUTEst SIF files are not redistributed here; their original distribution terms apply.

Set `MASTSIF` to the local directory containing the 17 definitions before running the LazyIPDCA, Ipopt or QPALM XL17 scripts. The exact historical LazyIPDCA shell protocol is preserved in `benchmarks/legacy_protocols/run_C_XL_v275_17_FINAL.sh`.

## LCP6--LCP13

`benchmarks/lcp/lcp_families.py` contains the generated problem families and the certificate used by the article. `benchmarks/manifests/lcp72.csv` contains all 72 dimensions. `benchmarks/manifests/lcp67_ipopt.csv` is the 67-row Ipopt matched subset.

QPALM tolerance tightening is implemented by `run_qpalm_certificate_matched.py`; Ipopt tolerance tightening for the matched subset is implemented by `run_ipopt_certificate_matched.py`. Selected-run times, rather than cumulative tightening time, are used in the article tables.

The certificate requires, with `w=Lx-g`, that nonnegativity violation, `||x .* w||_inf`, and `|x^T w|` are all at most `1e-6`.

## Sensitivity and timing

The full one-factor sensitivity grid is local to five representative instances. The scalar-decrease period `T_rho`, the only moderate control that changed trajectories in that grid, is additionally evaluated at 1, 2 and 5 on five more instances, yielding a ten-instance, 30-run study. The full parameter evidence contains 85 distinct runs.

Timing reproducibility uses ten repetitions on each of the original five instances. One-factor wall times are not used to rank parameter settings; repeated timing results are summarized by medians and coefficients of variation.

## Post-processing

`postprocess/summarize_reference.py` prints the reference numerical manifest. `postprocess/make_performance_profile.py` generates the quality-filtered performance profile from processed ratios. The processed numerical evidence used by the article is under `results/reference/` and `results/article_numeric_materials/`.

## Repository audit

Before publishing or tagging a release:

```bash
python scripts/audit_public_repo.py
bash scripts/preflight.sh
```

The audit also rejects build/backup artifacts and excluded authorship-assistance terminology from the public tree.
