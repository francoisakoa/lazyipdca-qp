#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,os,re,time,traceback
from pathlib import Path
import numpy as np
import scipy.sparse as sp

FAMILIES=('LCP7','LCP8','LCP10','LCP11','LCP12')
DIMS=(1_000_000,2_000_000,3_000_000,4_000_000,5_000_000)

def generator(fam,n):
    one=np.ones(n-1,dtype=float)
    if fam in ('LCP7','LCP11'):
        d=np.full(n,4.0); 
        if fam=='LCP11': d[0]=-4.0
        L=sp.diags((one,d,-2.0*one),(-1,0,1),shape=(n,n),format='csc')
        g=np.ones(n)
        if fam=='LCP11': g[0]=0.0; g[1:]=-1.0
    elif fam in ('LCP8','LCP12'):
        d=np.full(n,4.0)
        if fam=='LCP12': d[0]=-4.0
        L=sp.diags((-one,d,-one),(-1,0,1),shape=(n,n),format='csc')
        g=np.ones(n)
        if fam=='LCP12': g[:2]=0.0; g[2:]=-1.0
    elif fam=='LCP10':
        d=np.arange(1,n+1,dtype=float)/float(n)
        L=sp.diags(d,0,format='csc'); g=np.ones(n)
    else: raise ValueError(fam)
    Q=(L+L.T).tocsc(); q=-g
    I=sp.eye(n,format='csc')
    A=sp.vstack((I,L),format='csc')
    l=np.concatenate((np.zeros(n),g))
    u=np.full(2*n,np.inf)
    x0=np.zeros(n)
    return L,g,Q,q,A,l,u,x0

def convex_info(fam,n):
    if fam=='LCP7': return True, 8.0-2.0*math.cos(math.pi/(n+1)), 'exact_tridiagonal'
    if fam=='LCP8': return True, 8.0-4.0*math.cos(math.pi/(n+1)), 'exact_tridiagonal'
    if fam=='LCP10': return True, 2.0/n, 'exact_diagonal'
    if fam in ('LCP11','LCP12'): return False,-8.0,'negative_rayleigh_e1'
    raise ValueError(fam)

def lcp_metrics(L,g,x,tol):
    w=np.asarray(L@x-g).ravel()
    pv=max(0.0, float(-np.min(x)), float(-np.min(w)))
    prod=np.abs(x*w); comp=float(np.max(prod)) if prod.size else 0.0
    gap=float(abs(np.dot(x,w)))
    nat=float(np.max(np.abs(np.minimum(x,w)))) if x.size else 0.0
    cert=bool(pv<=tol and comp<=tol and gap<=tol)
    return w,pv,comp,gap,nat,cert

def kkt_metrics(Q,q,A,l,x,y,tol):
    z=np.asarray(A@x).ravel(); slack=z-l
    feas=max(0.0,float(np.max(-slack)))
    grad=np.asarray(Q@x+q).ravel(); aty=np.asarray(A.T@y).ravel(); r=grad+aty
    stat_abs=float(np.max(np.abs(r)))
    scale=1.0+max(float(np.max(np.abs(q))),float(np.max(np.abs(Q@x))),float(np.max(np.abs(aty))))
    stat_rel=stat_abs/scale
    # lower-only convention: native interval multiplier should be <= 0
    dual_sign=max(0.0,float(np.max(y)))
    lam=np.maximum(-y,0.0)
    comp_abs=float(np.max(lam*np.maximum(slack,0.0)))
    comp_rel=comp_abs/(1.0+max(1.0,float(np.max(np.abs(l))))*max(1.0,float(np.max(np.abs(y)))))
    ok=bool(feas<=tol and stat_rel<=tol and comp_rel<=tol and dual_sign<=tol)
    return dict(feas_abs=feas,stat_abs=stat_abs,stat_rel=stat_rel,comp_abs=comp_abs,comp_rel=comp_rel,dual_sign_abs=dual_sign,kkt1_ok=ok)

def run_qpalm(Q,q,A,l,u,x0,args):
    import qpalm
    n=Q.shape[0]; m=A.shape[0]
    data=qpalm.Data(n,m); data.Q=Q; data.q=q; data.A=A
    data.bmin=l; data.bmax=np.where(np.isposinf(u),1e20,u)
    st=qpalm.Settings(); st.max_iter=args.max_iter; st.eps_abs=args.tol; st.eps_rel=0.0; st.eps_prim_inf=args.tol; st.eps_dual_inf=args.tol; st.time_limit=args.time_limit; st.nonconvex=1; st.verbose=0
    solver=qpalm.Solver(data,st)
    try: solver.warm_start(x0,np.zeros(m))
    except Exception: pass
    t=time.perf_counter(); solver.solve(); wall=time.perf_counter()-t
    x=np.asarray(solver.solution.x).ravel().copy(); y=np.asarray(solver.solution.y).ravel().copy(); info=solver.info
    it=next((getattr(info,k) for k in ('iter','iters','iterations','iter_out') if hasattr(info,k)),None)
    return x,y,str(getattr(info,'status','unknown')),int(it) if it is not None else None,wall

def run_osqp(Q,q,A,l,u,x0,args):
    import osqp
    s=osqp.OSQP(); t=time.perf_counter()
    s.setup(P=sp.triu(Q,format='csc'),q=q,A=A,l=l,u=u,verbose=False,max_iter=args.max_iter,eps_abs=args.tol,eps_rel=0.0,eps_prim_inf=args.tol,eps_dual_inf=args.tol,time_limit=args.time_limit,polishing=True,scaled_termination=False,warm_starting=True)
    try:s.warm_start(x=x0)
    except Exception:pass
    r=s.solve(raise_error=False); wall=time.perf_counter()-t
    x=None if r.x is None else np.asarray(r.x).ravel().copy(); y=None if r.y is None else np.asarray(r.y).ravel().copy()
    return x,y,str(r.info.status),int(r.info.iter),wall

def js(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
FIELDS=['solver','family','n','status','iterations','wall_time_s','objective_qp','gap_signed','lcp_primal_violation','lcp_comp_inf','lcp_gap_abs','lcp_natural_inf','lcp_cert','feas_abs','stat_rel','comp_rel','dual_sign_abs','kkt1_ok','convex_qp','lambda_min_info','comparable','note']
def write_result(path,r):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps({k:js(v) for k,v in r.items()},indent=2,sort_keys=True)+'\n')
def aggregate(out):
    rs=[]
    for p in out.glob('*/*/result.json'):
        try: rs.append(json.loads(p.read_text()))
        except: pass
    order={f:i for i,f in enumerate(FAMILIES)}; rs.sort(key=lambda r:(order.get(r['family'],99),int(r['n']),r['solver']))
    with (out/'results_lcp_xl_qpalm_osqp.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore'); w.writeheader(); w.writerows(rs)
    return rs

def main():
    p=argparse.ArgumentParser(); p.add_argument('--outdir',required=True); p.add_argument('--solvers',default='qpalm,osqp'); p.add_argument('--max-iter',type=int,default=50000); p.add_argument('--time-limit',type=float,default=36000); p.add_argument('--tol',type=float,default=1e-6); p.add_argument('--only',action='append',default=[]); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); solvers=[s.strip() for s in a.solvers.split(',') if s.strip()]
    cases=[(f,n) for f in FAMILIES for n in DIMS]
    if a.only:
        cases=[]
        seen=set()
        for z in a.only:
            f,ns=z.split(':',1); case=(f.upper(),int(ns.replace(',','')))
            if case[0] not in FAMILIES: raise SystemExit(f'unsupported XL family: {case[0]}')
            if case not in seen: cases.append(case); seen.add(case)
    print('===== LCP XL / QPALM + OSQP ====='); print('outdir=',out); print('cases=',len(cases),'max_iter=',a.max_iter,'time_limit=',a.time_limit,'tol=',a.tol); print('solvers=',solvers,flush=True)
    for j,(fam,n) in enumerate(cases,1):
        print(f'\n===== [{j:02d}/{len(cases):02d}] {fam} n={n} =====',flush=True)
        L,g,Q,q,A,l,u,x0=generator(fam,n); psd,lmin,lmethod=convex_info(fam,n)
        print(f'[QP] n={n} nnzL={L.nnz} nnzQ={Q.nnz} nnzA={A.nnz} convex={int(psd)} lambda_info={lmin:.6e} ({lmethod})',flush=True)
        for solver in solvers:
            rp=out/solver/f'{fam}_n{n}'/'result.json'
            if a.resume and rp.exists(): print(f'[RESUME] {solver} {fam}_n{n}',flush=True); continue
            base=dict(solver=solver,family=fam,n=n,convex_qp=psd,lambda_min_info=lmin,comparable=True,note='')
            try:
                if solver=='osqp' and not psd:
                    r={**base,'status':'unsupported_nonconvex','iterations':0,'wall_time_s':0.0,'objective_qp':math.nan,'gap_signed':math.nan,'lcp_primal_violation':math.nan,'lcp_comp_inf':math.nan,'lcp_gap_abs':math.nan,'lcp_natural_inf':math.nan,'lcp_cert':False,'feas_abs':math.nan,'stat_rel':math.nan,'comp_rel':math.nan,'dual_sign_abs':math.nan,'kkt1_ok':False,'comparable':False,'note':'Q indefinite by e1 Rayleigh quotient; original QP not modified'}
                else:
                    if solver=='qpalm': x,y,status,it,wall=run_qpalm(Q,q,A,l,u,x0,a)
                    elif solver=='osqp': x,y,status,it,wall=run_osqp(Q,q,A,l,u,x0,a)
                    else: raise ValueError('unknown solver '+solver)
                    if x is None or y is None:
                        raise RuntimeError('solver returned no usable primal/dual point')
                    w,pv,lc,gap,nat,cert=lcp_metrics(L,g,x,a.tol); km=kkt_metrics(Q,q,A,l,x,y,a.tol)
                    obj=float(0.5*np.dot(x,Q@x)+np.dot(q,x)); sgap=float(np.dot(x,w))
                    r={**base,'status':status,'iterations':it,'wall_time_s':wall,'objective_qp':obj,'gap_signed':sgap,'lcp_primal_violation':pv,'lcp_comp_inf':lc,'lcp_gap_abs':gap,'lcp_natural_inf':nat,'lcp_cert':cert,**km,'note':'native solver dual; no solution vectors stored'}
                write_result(rp,r)
                print(f"[run] {solver:<5} {fam}_n{n:<8} status={str(r['status']):<24} gap={r['lcp_gap_abs'] if isinstance(r['lcp_gap_abs'],float) else math.nan:.3e} comp={r['lcp_comp_inf'] if isinstance(r['lcp_comp_inf'],float) else math.nan:.3e} LCP={int(r['lcp_cert'])} stat={r['stat_rel'] if isinstance(r['stat_rel'],float) else math.nan:.3e} KKT={int(r['kkt1_ok'])} wall={r['wall_time_s']:.3f}s cmp={int(r['comparable'])}",flush=True)
            except Exception as e:
                r={**base,'status':'exception','iterations':None,'wall_time_s':math.nan,'objective_qp':math.nan,'gap_signed':math.nan,'lcp_primal_violation':math.nan,'lcp_comp_inf':math.nan,'lcp_gap_abs':math.nan,'lcp_natural_inf':math.nan,'lcp_cert':False,'feas_abs':math.nan,'stat_rel':math.nan,'comp_rel':math.nan,'dual_sign_abs':math.nan,'kkt1_ok':False,'note':f'{type(e).__name__}: {e}'}; write_result(rp,r); (rp.parent/'exception.txt').write_text(traceback.format_exc()); print('[FAIL]',solver,fam,n,e,flush=True)
        del L,g,Q,q,A,l,u,x0
    rs=aggregate(out); print('\n===== SUMMARY =====')
    for s in solvers:
        q=[r for r in rs if r['solver']==s]; print(f"{s:<5}: rows={len(q):2d}/25 comparable={sum(bool(r.get('comparable')) for r in q):2d}/25 LCP_CERT={sum(bool(r.get('lcp_cert')) for r in q):2d}/25 KKT1={sum(bool(r.get('kkt1_ok')) for r in q):2d}/25 wall={sum(float(r.get('wall_time_s') or 0) for r in q):.3f}s")
    print('CSV =',out/'results_lcp_xl_qpalm_osqp.csv')
if __name__=='__main__': main()
