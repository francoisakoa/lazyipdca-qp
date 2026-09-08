#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--input", default="/data/projects/IPDCA_QP/ARTICLE_NUMERIC_FINAL_C381_MATCHED/performance_profile_ratios_3solver.csv")
ap.add_argument("--out", default="performance_profile_3solver.pdf")
args=ap.parse_args()

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit("matplotlib is required in the environment used to make the figure: "+str(e))

rows=[]
with open(args.input,newline="",errors="replace") as f:
    rows=list(csv.DictReader(f))

solvers=[("lazy","LazyIPDCA"),("ipopt","Ipopt"),("qpalm","QPALM")]
ratios={k:[] for k,_ in solvers}
for r in rows:
    for k,_ in solvers:
        try:
            x=float(r[f"{k}_ratio"])
        except:
            x=math.inf
        ratios[k].append(x)

finite=sorted({x for xs in ratios.values() for x in xs if math.isfinite(x) and x>=1})
if not finite:
    raise SystemExit("No finite performance ratios found.")

taus=[1.0]+finite
n=len(rows)

for k,label in solvers:
    ys=[sum(math.isfinite(x) and x<=tau for x in ratios[k])/n for tau in taus]
    plt.step(taus,ys,where="post",label=label)

plt.xscale("log")
plt.xlabel(r"Performance ratio $\tau$")
plt.ylabel(r"Fraction of instances with $r_{p,s}\leq\tau$")
plt.ylim(0,1.02)
plt.grid(True,which="both",alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(args.out,bbox_inches="tight")
print(args.out)
