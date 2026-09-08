#!/usr/bin/env python3
"""Certificate-matched Ipopt benchmark for the 67 LCP rows used in the article."""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from lcp_families import generate, certificate

class QPObj:
    def __init__(self,Q,q,A):
        self.Q=sp.csc_matrix(Q); self.q=np.asarray(q,dtype=float); self.A=sp.coo_matrix(A)
        L=sp.tril(self.Q,format='coo')
        self.hr=np.asarray(L.row,dtype=np.int32); self.hc=np.asarray(L.col,dtype=np.int32); self.hv=np.asarray(L.data,dtype=float)
        self.jr=np.asarray(self.A.row,dtype=np.int32); self.jc=np.asarray(self.A.col,dtype=np.int32); self.jv=np.asarray(self.A.data,dtype=float)
    def objective(self,x): return float(0.5*np.dot(x,self.Q@x)+np.dot(self.q,x))
    def gradient(self,x): return np.asarray(self.Q@x+self.q,dtype=float)
    def constraints(self,x): return np.asarray(self.A@x,dtype=float)
    def jacobian(self,x): return self.jv
    def jacobianstructure(self): return self.jr,self.jc
    def hessian(self,x,lagrange,obj_factor): return float(obj_factor)*self.hv
    def hessianstructure(self): return self.hr,self.hc

def solve_ipopt(L,g,Q,q,eps,max_iter,time_limit,print_level):
    import cyipopt
    n=len(g); A=sp.csc_matrix(L); obj=QPObj(Q,q,A)
    lb=np.zeros(n); ub=np.full(n,1e19); cl=np.asarray(g,dtype=float); cu=np.full(n,1e19)
    nlp=cyipopt.Problem(n=n,m=n,problem_obj=obj,lb=lb,ub=ub,cl=cl,cu=cu)
    for k,v in {'tol':eps,'constr_viol_tol':eps,'compl_inf_tol':eps,'dual_inf_tol':eps,'max_iter':max_iter,'linear_solver':'mumps','nlp_scaling_method':'gradient-based','mu_strategy':'monotone','acceptable_iter':0,'print_level':print_level}.items():
        nlp.add_option(k,v)
    try: nlp.add_option('max_wall_time',float(time_limit))
    except Exception: pass
    try: nlp.add_option('max_cpu_time',float(time_limit))
    except Exception: pass
    x0=np.maximum(1.0,np.ones(n))
    t=time.perf_counter(); x,info=nlp.solve(x0); wall=time.perf_counter()-t
    return np.asarray(x,dtype=float),str(info.get('status_msg',info.get('status',''))),int(info.get('iter_count',info.get('iterations',0)) or 0),wall

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default=str(here.parent/'manifests/lcp67_ipopt.csv')); ap.add_argument('--outdir',required=True); ap.add_argument('--max-iter',type=int,default=3000); ap.add_argument('--time-limit',type=float,default=36000); ap.add_argument('--cert-tol',type=float,default=1e-6); ap.add_argument('--resume',action='store_true'); ap.add_argument('--print-level',type=int,default=0); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); rows=[]
    for j,c in enumerate(csv.DictReader(open(args.manifest)),1):
        fam=c['family']; n=int(c['n']); rp=out/f'{fam}_n{n}.json'
        if args.resume and rp.exists(): rows.append(json.loads(rp.read_text())); continue
        L,g,Q,q=generate(fam,n); selected=None
        for eps in (1e-8,1e-10,1e-12,1e-13):
            x,status,it,wall=solve_ipopt(L,g,Q,q,eps,args.max_iter,args.time_limit,args.print_level)
            cert=certificate(L,g,x,args.cert_tol)
            selected={'family':fam,'n':n,'solver_tol':eps,'status':status,'iterations':it,'selected_wall_s':wall,**cert}
            print(f'[{j:02d}] {fam} n={n} tol={eps:g} cert={int(cert["certified"])} gap={cert["gap_abs"]:.3e} wall={wall:.3f}s',flush=True)
            if cert['certified']: break
        rp.write_text(json.dumps(selected,indent=2)+'\n'); rows.append(selected)
    fields=['family','n','solver_tol','status','iterations','selected_wall_s','primal_violation','comp_inf','gap_abs','certified']
    with open(out/'lcp67_ipopt_certificate_matched.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('certified',sum(bool(r['certified']) for r in rows),'/',len(rows))
if __name__=='__main__': main()
