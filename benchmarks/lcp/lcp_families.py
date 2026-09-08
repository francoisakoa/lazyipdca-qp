#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp

def generate(family,n):
    family=family.upper(); n=int(n); one=np.ones(n-1)
    if family=='LCP6':
        idx=np.arange(1,n+1,dtype=float); L=4*np.minimum.outer(idx,idx)-2; np.fill_diagonal(L,4*idx-3); L=sp.csc_matrix(L); g=np.ones(n)
    elif family in ('LCP7','LCP11'):
        d=np.full(n,4.0); d[0]=-4.0 if family=='LCP11' else d[0]; L=sp.diags((one,d,-2*one),(-1,0,1),shape=(n,n),format='csc'); g=np.ones(n); 
        if family=='LCP11': g[0]=0.0; g[1:]=-1.0
    elif family in ('LCP8','LCP12'):
        d=np.full(n,4.0); d[0]=-4.0 if family=='LCP12' else d[0]; off=-one; L=sp.diags((off,d,off),(-1,0,1),shape=(n,n),format='csc'); g=np.ones(n)
        if family=='LCP12': g[:2]=0.0; g[2:]=-1.0
    elif family in ('LCP9','LCP13'):
        L=np.triu(np.full((n,n),2.0),1)+np.eye(n); g=np.ones(n)
        if family=='LCP13': L[-1,-1]=-1.0; g[-1]=0.0
        L=sp.csc_matrix(L)
    elif family=='LCP10': L=sp.diags(np.arange(1,n+1,dtype=float)/n,0,format='csc'); g=np.ones(n)
    else: raise ValueError(family)
    return L,g,(L+L.T).tocsc(),-g

def certificate(L,g,x,tol=1e-6):
    x=np.asarray(x,float).ravel(); w=np.asarray(L@x-g).ravel(); primal=max(0.0,float(-x.min()),float(-w.min())); comp=float(np.max(np.abs(x*w))) if x.size else 0.0; gap=float(abs(np.dot(x,w))); return {'primal_violation':primal,'comp_inf':comp,'gap_abs':gap,'certified':primal<=tol and comp<=tol and gap<=tol}
