#!/usr/bin/env python3
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from lcp_families import generate, certificate

def solve_qpalm(Q,q,A,l,u,x0,eps,max_iter,time_limit):
    import qpalm
    data=qpalm.Data(Q.shape[0],A.shape[0])
    data.Q=Q; data.q=q; data.A=A; data.bmin=l; data.bmax=np.where(np.isposinf(u),1e20,u)
    st=qpalm.Settings(); st.max_iter=max_iter; st.eps_abs=eps; st.eps_rel=0.0
    st.eps_prim_inf=eps; st.eps_dual_inf=eps; st.time_limit=time_limit; st.nonconvex=1; st.verbose=0
    solver=qpalm.Solver(data,st)
    try:
        solver.warm_start(x0,np.zeros(A.shape[0]))
    except Exception:
        pass
    t=time.perf_counter(); solver.solve(); wall=time.perf_counter()-t
    return np.asarray(solver.solution.x).ravel().copy(), str(solver.info.status), int(getattr(solver.info,'iter',getattr(solver.info,'iterations',0))), wall

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default=str(Path(__file__).resolve().parents[1]/'manifests/lcp72.csv'))
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--max-iter',type=int,default=50000)
    ap.add_argument('--time-limit',type=float,default=600)
    ap.add_argument('--cert-tol',type=float,default=1e-6)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); rows=[]
    for j,c in enumerate(csv.DictReader(open(args.manifest)),1):
        fam=c['family']; n=int(c['n']); rp=out/f'{fam}_n{n}.json'
        if args.resume and rp.exists():
            rows.append(json.load(open(rp))); continue
        L,g,Q,q=generate(fam,n)
        I=sp.eye(n,format='csc'); A=sp.vstack((I,L),format='csc')
        l=np.concatenate((np.zeros(n),g)); u=np.full(2*n,np.inf); x0=np.zeros(n); selected=None
        for eps in (1e-6,1e-8,1e-10,1e-12,1e-13):
            x,status,it,wall=solve_qpalm(Q,q,A,l,u,x0,eps,args.max_iter,args.time_limit)
            cert=certificate(L,g,x,args.cert_tol)
            selected={'family':fam,'n':n,'solver_eps':eps,'status':status,'iterations':it,'selected_wall_s':wall,**cert}
            print(f'[{j:02d}] {fam} n={n} eps={eps:g} cert={int(cert["certified"])} gap={cert["gap_abs"]:.3e} wall={wall:.3f}s',flush=True)
            if cert['certified']:
                break
        rp.write_text(json.dumps(selected,indent=2)+'\n')
        rows.append(selected)
    fields=['family','n','solver_eps','status','iterations','selected_wall_s','primal_violation','comp_inf','gap_abs','certified']
    with open(out/'lcp72_qpalm_certificate_matched.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('certified',sum(bool(r['certified']) for r in rows),'/',len(rows))

if __name__=='__main__':
    main()
