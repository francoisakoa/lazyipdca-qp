#!/usr/bin/env python3
import csv, math
from pathlib import Path

base=Path("/data/projects/IPDCA_QP")
files={
 "lazy":base/"CUTEST_37_ABCD_v275_20260824_123948/tables_37_abcd/table_C_37.csv",
 "qpalm":base/"CUTEST_37_QPALM_OSQP_MAX50000_20260831_090328/results_37_qpalm_osqp.csv",
 "ipopt":base/"cutest_ipopt_37_20260820_190515/ipopt_cutest37_summary_complete.csv",
}
for k,p in files.items():
    if not p.exists(): raise SystemExit(f"missing {k}: {p}")
print("All authoritative CUTEst37 inputs found.")

with open(files["lazy"],newline="",errors="replace") as f:
    rows=list(csv.DictReader(f))
tot=sum(float(r["time_s"]) for r in rows)
print("Lazy C total:",tot)
if abs(tot-381.251494576)>1e-6:
    raise SystemExit("Unexpected Lazy C total wall time.")
print("PASS")
