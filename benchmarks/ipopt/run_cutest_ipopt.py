#!/usr/bin/env python3
"""Run the article Ipopt protocol on the matched CUTEst37 or XL17 panel.

The script reuses the same PyCUTEst-to-QP extraction used by the QPALM
benchmarks, retains fixed variables, starts from CUTEst x0, and solves the
linear-constraint QP through cyipopt with a constant sparse Hessian/Jacobian.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, math, os, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import scipy.sparse as sp


def load_helper(path: Path):
    spec=importlib.util.spec_from_file_location("ipdca_qp_panel_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def finite_or(x, fallback):
    x=np.asarray(x,dtype=float).copy()
    x[np.isneginf(x)] = -fallback
    x[np.isposinf(x)] = fallback
    return x


class CyIpoptQP:
    def __init__(self,d):
        self.d=d
        self.J=sp.coo_matrix(d.J)
        qlower=sp.tril(d.Q,format="coo")
        self.hr=np.asarray(qlower.row,dtype=np.int32)
        self.hc=np.asarray(qlower.col,dtype=np.int32)
        self.hv=np.asarray(qlower.data,dtype=float)
        self.jr=np.asarray(self.J.row,dtype=np.int32)
        self.jc=np.asarray(self.J.col,dtype=np.int32)
        self.jv=np.asarray(self.J.data,dtype=float)
    def objective(self,x):
        return float(self.d.f0 + 0.5*np.dot(x,self.d.Q@x) + np.dot(self.d.q,x))
    def gradient(self,x):
        return np.asarray(self.d.Q@x + self.d.q,dtype=float)
    def constraints(self,x):
        return np.asarray(self.d.J@x,dtype=float)
    def jacobian(self,x):
        return self.jv
    def jacobianstructure(self):
        return self.jr,self.jc
    def hessian(self,x,lagrange,obj_factor):
        return float(obj_factor)*self.hv
    def hessianstructure(self):
        return self.hr,self.hc


def max_constraint_violation(d,x):
    vals=[]
    if d.m_original:
        c=np.asarray(d.J@x,dtype=float)
        if np.any(np.isfinite(d.cl)):
            vals.append(float(np.max(np.maximum(d.cl-c,0.0))))
        if np.any(np.isfinite(d.cu)):
            vals.append(float(np.max(np.maximum(c-d.cu,0.0))))
    if np.any(np.isfinite(d.bl)):
        vals.append(float(np.max(np.maximum(d.bl-x,0.0))))
    if np.any(np.isfinite(d.bu)):
        vals.append(float(np.max(np.maximum(x-d.bu,0.0))))
    return max(vals) if vals else 0.0


def native_kkt(d,x,info):
    y=np.asarray(info.get("mult_g",np.zeros(d.m_original)),dtype=float).reshape(-1)
    zl=np.asarray(info.get("mult_x_L",np.zeros(d.n)),dtype=float).reshape(-1)
    zu=np.asarray(info.get("mult_x_U",np.zeros(d.n)),dtype=float).reshape(-1)
    g=np.asarray(d.Q@x+d.q,dtype=float)
    stat_vec=g.copy()
    if d.m_original:
        stat_vec += np.asarray(d.J.T@y,dtype=float).reshape(-1)
    # Ipopt bound convention: grad L = grad f + J'g - z_L + z_U.
    stat_vec += -zl + zu
    scale=max(1.0,float(np.linalg.norm(g,np.inf)) if g.size else 0.0)
    stat_abs=float(np.linalg.norm(stat_vec,np.inf)) if stat_vec.size else 0.0
    stat_rel=stat_abs/scale
    comps=[]
    if d.m_original:
        c=np.asarray(d.J@x,dtype=float)
        for i,yi in enumerate(y):
            if yi < 0 and math.isfinite(d.cl[i]):
                comps.append(abs(yi*(c[i]-d.cl[i])))
            elif yi > 0 and math.isfinite(d.cu[i]):
                comps.append(abs(yi*(d.cu[i]-c[i])))
    for i in range(d.n):
        if math.isfinite(d.bl[i]): comps.append(abs(zl[i]*(x[i]-d.bl[i])))
        if math.isfinite(d.bu[i]): comps.append(abs(zu[i]*(d.bu[i]-x[i])))
    comp_abs=max(comps) if comps else 0.0
    comp_rel=comp_abs/max(1.0,abs(float(info.get("obj_val",0.0))))
    return stat_abs,stat_rel,comp_abs,comp_rel


def solve_one(d,args):
    import cyipopt
    obj=CyIpoptQP(d)
    lb=finite_or(d.bl,1e19); ub=finite_or(d.bu,1e19)
    cl=finite_or(d.cl,1e19); cu=finite_or(d.cu,1e19)
    nlp=cyipopt.Problem(n=d.n,m=d.m_original,problem_obj=obj,lb=lb,ub=ub,cl=cl,cu=cu)
    options={
        "tol":args.tol,
        "max_iter":args.max_iter,
        "linear_solver":"mumps",
        "nlp_scaling_method":"gradient-based",
        "mu_strategy":"monotone",
        "acceptable_iter":0,
        "print_level":args.print_level,
    }
    for k,v in options.items(): nlp.add_option(k,v)
    # Recent Ipopt supports both limits; keep CPU limit as fallback.
    try: nlp.add_option("max_wall_time",float(args.time_limit))
    except Exception: pass
    try: nlp.add_option("max_cpu_time",float(args.time_limit))
    except Exception: pass
    t=time.perf_counter(); x,info=nlp.solve(np.asarray(d.x0,dtype=float)); wall=time.perf_counter()-t
    x=np.asarray(x,dtype=float).reshape(-1)
    f=float(obj.objective(x)); feas=max_constraint_violation(d,x)
    stat_abs,stat_rel,comp_abs,comp_rel=native_kkt(d,x,info)
    return {
        "tag":d.tag,"problem":d.problem,"n":d.n,"m":d.m_original,
        "iterations":int(info.get("iter_count",info.get("iterations",0)) or 0),
        "objective":f,"dual_abs":stat_abs,"dual_rel":stat_rel,
        "feas_abs":feas,"comp_abs":comp_abs,"comp_rel":comp_rel,
        "wall_s":wall,"feasible_1e-6":int(feas<=1e-6),
        "exit":str(info.get("status_msg",info.get("status",""))),
        "status_code":int(info.get("status",0)) if str(info.get("status","")).lstrip('-').isdigit() else info.get("status"),
    },x


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--panel",choices=["37","xl17"],default="37")
    ap.add_argument("--manifest",type=Path,default=None)
    ap.add_argument("--outdir",type=Path,required=True)
    ap.add_argument("--only",action="append",default=[])
    ap.add_argument("--tol",type=float,default=1e-8)
    ap.add_argument("--max-iter",type=int,default=3000)
    ap.add_argument("--time-limit",type=float,default=36000.0)
    ap.add_argument("--print-level",type=int,default=0)
    ap.add_argument("--pycutest-cache",default=os.environ.get("PYCUTEST_CACHE",str(Path.home()/".cache/pycutest")))
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[2]
    if args.panel=="37":
        helper_path=repo/"benchmarks/cutest37/qpalm/bench37.py"
        manifest=args.manifest or repo/"benchmarks/manifests/cutest37.csv"
    else:
        helper_path=repo/"benchmarks/xl17/qpalm/bench_xl17.py"
        manifest=args.manifest or repo/"benchmarks/manifests/xl17.csv"
    helper=load_helper(helper_path)
    entries=helper.read_manifest(Path(manifest))
    canonical=list(helper.TAGS37)
    tags=args.only or canonical
    bad=[t for t in tags if t not in entries]
    if bad: raise SystemExit(f"manifest missing tags: {bad}")
    args.outdir.mkdir(parents=True,exist_ok=True)
    cache=Path(args.pycutest_cache).expanduser().resolve(); cache.mkdir(parents=True,exist_ok=True)
    os.environ["PYCUTEST_CACHE"]=str(cache)
    rows=[]
    for i,tg in enumerate(tags,1):
        jp=args.outdir/f"{tg}.json"
        if args.resume and jp.exists():
            rows.append(json.loads(jp.read_text())); print(f"[RESUME] {tg}"); continue
        print(f"===== [{i:02d}/{len(tags):02d}] {tg} =====",flush=True)
        d=helper.build_qp(entries[tg],f"ipopt_{args.panel}")
        rec,x=solve_one(d,args)
        jp.write_text(json.dumps(rec,indent=2)+"\n")
        rows.append(rec)
        print(f"[run] {tg:<24} obj={rec['objective']:.9e} feas={rec['feas_abs']:.3e} stat={rec['dual_rel']:.3e} wall={rec['wall_s']:.3f}s",flush=True)
    fields=["tag","problem","n","m","iterations","objective","dual_abs","feas_abs","comp_abs","dual_rel","comp_rel","wall_s","feasible_1e-6","exit","status_code"]
    name="ipopt_cutest37_summary_complete.csv" if args.panel=="37" else "ipopt_xl17_summary.csv"
    with open(args.outdir/name,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(args.outdir/name)

if __name__=="__main__": main()
