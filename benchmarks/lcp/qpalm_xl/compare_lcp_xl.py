#!/usr/bin/env python3
import argparse,csv
p=argparse.ArgumentParser(); p.add_argument('--solver-csv',required=True); p.add_argument('--lazy-csv',default='lazyipdca_lcp_xl_reference.csv'); p.add_argument('--out',default='lcp_xl_qpalm_osqp_vs_lazyipdca.csv');a=p.parse_args()
def R(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
lazy={(r['family'],int(r['n'])):r for r in R(a.lazy_csv)}; sol=R(a.solver_csv); out=[]
for r in sol:
 k=(r['family'],int(r['n'])); l=lazy[k]; rr=dict(r); rr.update(lazy_time_s=l['lazy_time_s'],lazy_gap_abs=l['lazy_gap_abs'],lazy_cert=l['lazy_cert']);
 try: rr['time_ratio_solver_over_lazy']=float(r['wall_time_s'])/float(l['lazy_time_s']) if r.get('comparable') in ('True','true','1',True) else ''
 except: rr['time_ratio_solver_over_lazy']=''
 out.append(rr)
fields=list(out[0].keys());
with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
print('CSV =',a.out)
