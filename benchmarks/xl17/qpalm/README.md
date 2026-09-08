# CUTEst XL17 / QPALM v1

QPALM runner for the canonical 17-instance large-scale panel used by LazyIPDCA.

- 8 historical NCVXBQP/NCVXQP at n=100000
- QPNBAND at n=100000
- STNQP1/2 at n=131073 via the existing local SIF copies (P=17)
- BLOWEYA/B/C at n=100002 via local SIF copies (N=50000)
- BLOCKQP1/2/3 at n=100010 via local SIF copies (N=50000, B=10)

The script starts from the original CUTEst x0, keeps fixed variables, uses QPALM nonconvex mode, and reports objective, feasibility, native-dual stationarity, complementarity, KKT1, iterations and wall time.

Default campaign settings: max_iter=50000, time_limit=36000 s, tol=1e-6, one BLAS/OpenMP thread.
