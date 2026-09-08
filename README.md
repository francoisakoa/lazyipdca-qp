# IPDCA-QP / LazyIPDCA

Reproducibility repository for **Adaptive Low-Rank DC Regularization for Large-Scale Nonconvex Quadratic Programming**.

Repository: https://github.com/francoisakoa/ipdca-qp

LazyIPDCA solves linearly constrained nonconvex quadratic programs using event-driven low-rank DC regularization, a residual scalar safeguard, and a primal-dual interior-point method with bordered KKT systems.

## Quick start on Ubuntu / WSL2

```bash
bash scripts/install_system_deps_ubuntu.sh
conda env create -f environment.yml
conda activate ipdca-qp
bash scripts/install_cutest.sh
source "$HOME/opt/cutest/env.sh"
bash scripts/build_cutest.sh
bash scripts/run_tests.sh
bash scripts/preflight.sh
```

## Reproduce the article campaigns

### LazyIPDCA: CUTEst37 A--D and scalar ablation

```bash
conda activate ipdca-qp
source "$HOME/opt/cutest/env.sh"
bash benchmarks/cutest37/run_lazyipdca_abcd.sh
bash benchmarks/cutest37/run_scalar_ablation.sh
```

### Ipopt and QPALM on CUTEst37

```bash
conda env create -f environment-benchmarks.yml
conda activate ipdca-qp-bench
source "$HOME/opt/cutest/env.sh"
bash benchmarks/ipopt/run_cutest37.sh
bash benchmarks/cutest37/qpalm/run_37.sh
```

### XL17

The official CUTEst/MASTSIF definitions are not redistributed. The eight parametric extensions used by the article must be prepared locally from their official SIF definitions as described in `docs/REPRODUCIBILITY.md`.

```bash
conda activate ipdca-qp
source "$HOME/opt/cutest/env.sh"
bash benchmarks/legacy_protocols/run_C_XL_v275_17_FINAL.sh

conda activate ipdca-qp-bench
bash benchmarks/ipopt/run_xl17.sh
bash benchmarks/xl17/qpalm/run_xl17_qpalm.sh
```

### Generated LCP families

The repository includes the LCP6--LCP13 generators and the original-problem certificate.

```bash
conda activate ipdca-qp-bench
python benchmarks/lcp/run_qpalm_certificate_matched.py \
  --outdir results/runs/lcp72_qpalm --resume
bash benchmarks/lcp/run_ipopt67.sh
```

The validated LazyIPDCA LCP results and certificate summaries used in the article are provided under `results/reference/`; the generic LCP definitions and certificate are in `benchmarks/lcp/lcp_families.py`.

### Sensitivity and timing

```bash
conda activate ipdca-qp
source "$HOME/opt/cutest/env.sh"
bash benchmarks/sensitivity/run_rho_trial_10instances.sh
REPS=10 bash benchmarks/sensitivity/run_timing_repetitions.sh
```

The full one-factor sensitivity grid is local to five instances; only the trajectory-sensitive `T_rho` axis is extended to ten instances.

## Reference numerical values

```bash
python postprocess/summarize_reference.py
```

The authoritative configuration-C CUTEst37 total is `381.251494576` seconds.

See `docs/REPRODUCIBILITY.md`, `docs/NUMERICAL_RESULTS.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`.
