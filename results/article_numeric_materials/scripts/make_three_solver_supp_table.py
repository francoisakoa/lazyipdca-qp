#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--input", default="/data/projects/IPDCA_QP/ARTICLE_NUMERIC_FINAL_C381_MATCHED/three_solver_37_matched.csv")
ap.add_argument("--out", default="table_three_solver_37.tex")
args=ap.parse_args()

with open(args.input,newline="",errors="replace") as f:
    rows=list(csv.DictReader(f))

def esc(s): return str(s).replace("_",r"\_")
def fnum(x):
    try:
        v=float(x)
        if not math.isfinite(v): return "--"
        return f"{v:.6g}"
    except: return "--"

lines=[]
lines += [r"\begin{longtable}{lrrrrrr}", r"\caption{Instance-level CUTEst37 comparison.}\label{tab:three-solver-37}\\",
          r"\toprule", r"Instance & $f_L$ & $t_L$ & $f_I$ & $t_I$ & $f_Q$ & $t_Q$\\", r"\midrule", r"\endfirsthead",
          r"\toprule", r"Instance & $f_L$ & $t_L$ & $f_I$ & $t_I$ & $f_Q$ & $t_Q$\\", r"\midrule", r"\endhead"]
for r in rows:
    lines.append(
        f"{esc(r['instance'])} & {fnum(r.get('lazy_obj'))} & {fnum(r.get('lazy_time'))} & "
        f"{fnum(r.get('ipopt_obj'))} & {fnum(r.get('ipopt_time'))} & "
        f"{fnum(r.get('qpalm_obj'))} & {fnum(r.get('qpalm_time'))} \\\\"
    )
lines += [r"\bottomrule", r"\end{longtable}"]
Path(args.out).write_text("\n".join(lines)+"\n")
print(args.out)
