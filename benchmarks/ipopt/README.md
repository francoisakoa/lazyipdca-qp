# Ipopt comparator

The CUTEst scripts implement the article protocol with Ipopt 3.14.19 through cyipopt 1.7.0, MUMPS, the original CUTEst starting point, fixed variables retained, gradient-based scaling, monotone barrier update, no acceptable-level early termination, a 3000-iteration safeguard and a 36000-second limit.

```bash
conda activate ipdca-qp-bench
source "$HOME/opt/cutest/env.sh"
bash benchmarks/ipopt/run_cutest37.sh
```

For the XL17 panel, set `MASTSIF` to a directory containing the official SIF files plus the eight documented parametric extensions before running `run_xl17.sh`. The repository does not redistribute CUTEst SIF files.
