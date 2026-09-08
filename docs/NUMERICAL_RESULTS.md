# Numerical reference summary

| Solver | Feasible | KKT1 | Total wall time (s) | Ratio to LazyIPDCA | Median s/iteration |
|---|---:|---:|---:|---:|---:|
| LazyIPDCA C | 37/37 | 27/37 | 381.251 | 1.00 | 8.719e-3 |
| Ipopt | 37/37 | 28/37 | 3428.100 | 8.99 | 2.301e-2 |
| QPALM | 37/37 | 35/37 | 6412.156 | 16.82 | 9.270e-4 |

Objective partitions: LazyIPDCA/Ipopt/tie = 21/3/13; LazyIPDCA/QPALM/tie = 20/3/14. Large-scale feasible returns: 17/16/11 for LazyIPDCA/Ipopt/QPALM. Generated LCP certification: LazyIPDCA 72/72, QPALM 71/72, and Ipopt 66/67 on its matched subset.
