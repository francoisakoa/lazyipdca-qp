#!/usr/bin/env python3
import csv, os, sys
from pathlib import Path

root=Path('/data/projects/IPDCA_QP')
kit=Path(__file__).resolve().parent
xlref=root/'CUTEST_C_XL_v275_20260825_141918'
mastsif=xlref/'MASTSIF_XL_LOCAL'
manifest=kit/'manifest_xl17.csv'

print('===== PREFLIGHT QPALM CUTEST XL17 =====')
print('python      =', sys.executable)
print('MASTSIF     =', mastsif)
print('manifest    =', manifest)

for mod in ['numpy','scipy','pycutest','qpalm']:
    try:
        m=__import__(mod)
        print(f'[PASS] {mod:<8} version={getattr(m,"__version__","unknown")}')
    except Exception as e:
        print(f'[FAIL] {mod}: {e}')
        raise SystemExit(2)

required=['STNQP1','STNQP2','BLOWEYA','BLOWEYB','BLOWEYC','BLOCKQP1','BLOCKQP2','BLOCKQP3']
for name in required:
    p=mastsif/f'{name}.SIF'
    if not p.exists():
        print('[FAIL] missing', p)
        raise SystemExit(2)
print('[PASS] 8/8 local parametric SIF files present')

rows=list(csv.DictReader(manifest.open()))
print(f'[PASS] manifest rows={len(rows)}/17')
if len(rows)!=17:
    raise SystemExit(2)
print('[PASS] preflight complete')
