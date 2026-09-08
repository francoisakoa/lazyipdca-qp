#!/usr/bin/env python3
import argparse, json, subprocess, time
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument("--binary",required=True)
ap.add_argument("--arm",required=True)
ap.add_argument("--tag",required=True)
ap.add_argument("--outdir",required=True)
ap.add_argument("--log",required=True)
ap.add_argument("--record",required=True)
ap.add_argument("args",nargs=argparse.REMAINDER)
a=ap.parse_args()
solver_args=list(a.args)
if solver_args and solver_args[0]=="--":
    solver_args=solver_args[1:]
if not solver_args:
    raise SystemExit("[FAIL] empty solver argument vector")
out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
log=Path(a.log); log.parent.mkdir(parents=True,exist_ok=True)
t0=time.perf_counter()
with log.open("w") as fh:
    p=subprocess.run([a.binary]+solver_args,stdout=fh,stderr=subprocess.STDOUT)
wall=time.perf_counter()-t0
js=list(out.rglob(f"{a.tag}_raw.json"))
if not js:
    tail="\n".join(log.read_text(errors="replace").splitlines()[-25:])
    raise SystemExit(f"[FAIL] missing JSON for {a.arm}/{a.tag}; rc={p.returncode}; log={log}\n----- log tail -----\n{tail}")
jpath=js[0]
Path(a.record).write_text(json.dumps({"arm":a.arm,"tag":a.tag,"rc":p.returncode,"wall_sec":wall,"json":str(jpath),"log":str(log)})+"\n")
print(f"[run] {a.arm:8s} {a.tag:24s} rc={p.returncode:2d} wall={wall:9.3f}s")
