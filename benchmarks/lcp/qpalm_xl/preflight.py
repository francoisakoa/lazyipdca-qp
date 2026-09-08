#!/usr/bin/env python3
import importlib,os
print('===== PREFLIGHT LCP XL / QPALM / OSQP =====')
for n in ('numpy','scipy','qpalm','osqp'):
 try:
  m=importlib.import_module(n); print(f'[PASS] {n:<8} {getattr(m,"__version__","unknown")}')
 except Exception as e: print(f'[MISS] {n:<8} {e}')
print('OMP_NUM_THREADS=',os.getenv('OMP_NUM_THREADS','(set by run_lcp_xl.sh)'))
