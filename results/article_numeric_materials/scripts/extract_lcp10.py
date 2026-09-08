#!/usr/bin/env python3
import argparse, csv, json, re, math, subprocess
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--base", default="/data/projects/IPDCA_QP")
ap.add_argument("--out", default="lcp10_discovered.csv")
args=ap.parse_args()
base=Path(args.base)

def num(x):
    try:
        v=float(x); return v if math.isfinite(v) else None
    except: return None

def first(d,names):
    for k in names:
        if isinstance(d,dict) and k in d and d[k] not in (None,""):
            return d[k]
    return None

rows=[]
# Prefer ripgrep for content search; fallback to filename search.
try:
    p=subprocess.run(["rg","-l","-i","LCP10",str(base),"-g","*.csv","-g","*.json"],capture_output=True,text=True,check=False)
    files=[Path(x) for x in p.stdout.splitlines() if x.strip()]
except:
    files=list(base.rglob("*LCP10*.json"))+list(base.rglob("*LCP10*.csv"))

for fp in files:
    low=str(fp).lower()
    solver="Ipopt" if "ipopt" in low else ("QPALM" if "qpalm" in low else "LazyIPDCA_or_other")
    if fp.suffix.lower()==".json":
        if "lcp10" not in low: continue
        try: d=json.loads(fp.read_text(errors="replace"))
        except: continue
        if not isinstance(d,dict): continue
        m=re.search(r"LCP10[_-]n[_-]?(\d+)",str(fp),re.I)
        n=num(first(d,["n","dimension","size"])) or (int(m.group(1)) if m else None)
        rows.append({
            "solver":solver,"source":str(fp),"n":n,
            "objective":num(first(d,["objective","obj","final_objective"])),
            "feas":num(first(d,["feas","feasibility","feas_abs","primal_violation"])),
            "stat":num(first(d,["stat","stat_rel","kkt_stat_rel","stationarity"])),
            "comp":num(first(d,["comp","comp_rel","kkt_comp_rel","complementarity","comp_inf"])),
            "iterations":num(first(d,["iterations","iters","iter"])),
            "factorizations":num(first(d,["n_factorizations","factorizations","kkt_factorizations"])),
            "lanczos_calls":num(first(d,["n_lanczos_calls","lanczos_calls"])),
            "time_s":num(first(d,["time_s","wall_time_s","wall","time","elapsed_seconds"])),
            "status":str(first(d,["status","status_name","message"]) or "")
        })
    elif fp.suffix.lower()==".csv":
        try:
            with open(fp,newline="",errors="replace") as f:
                rr=csv.DictReader(f); data=list(rr)
        except: continue
        for d in data:
            blob=" ".join(str(v) for v in d.values())
            if "lcp10" not in blob.lower(): continue
            m=re.search(r"LCP10[_-]n[_-]?(\d+)",blob,re.I)
            n=num(first(d,["n","dimension","size"])) or (int(m.group(1)) if m else None)
            rows.append({
                "solver":solver,"source":str(fp),"n":n,
                "objective":num(first(d,["objective","obj","final_objective"])),
                "feas":num(first(d,["feas","feasibility","feas_abs","primal_violation"])),
                "stat":num(first(d,["stat","stat_rel","kkt_stat_rel","stationarity"])),
                "comp":num(first(d,["comp","comp_rel","kkt_comp_rel","complementarity","comp_inf"])),
                "iterations":num(first(d,["iterations","iters","iter"])),
                "factorizations":num(first(d,["n_factorizations","factorizations","kkt_factorizations"])),
                "lanczos_calls":num(first(d,["n_lanczos_calls","lanczos_calls"])),
                "time_s":num(first(d,["time_s","wall_time_s","wall","time","elapsed_seconds"])),
                "status":str(first(d,["status","status_name","message"]) or "")
            })

fields=["solver","source","n","objective","feas","stat","comp","iterations","factorizations","lanczos_calls","time_s","status"]
with open(args.out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)

print(f"{len(rows)} records -> {args.out}")
