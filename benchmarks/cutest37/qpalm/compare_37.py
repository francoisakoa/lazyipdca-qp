#!/usr/bin/env python3
import argparse,csv,math
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--qpalm-csv',required=True); p.add_argument('--lazy-csv',default=str(Path(__file__).with_name('lazyipdca_c5_reference_37.csv'))); p.add_argument('--out',default='qpalm_vs_lazyipdca_37.csv'); a=p.parse_args()
def rows(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
L={r['tag']:r for r in rows(a.lazy_csv)}
Q={r['tag']:r for r in rows(a.qpalm_csv) if r.get('solver','qpalm')=='qpalm'}
out=[]; c={'LazyIPDCA':0,'QPALM':0,'tie':0,'not_comparable':0}
for tag,l in L.items():
 q=Q.get(tag)
 if not q: out.append({'tag':tag,'verdict':'missing_qpalm'}); c['not_comparable']+=1; continue
 nL=int(float(l['n'])); mL=int(float(l['m'])); nQ=int(float(q['n'])); mQ=int(float(q['m']))
 aligned=(nL==nQ and mL==mQ)
 lf=float(l['objective']); qf=float(q['objective']); lfe=float(l['feas']); qfe=float(q['feas_abs'])
 if not aligned or lfe>1e-6 or qfe>1e-6:
  verdict='not_comparable'; c['not_comparable']+=1
 else:
  best=min(lf,qf); gl=abs(lf-best)/max(1,abs(best)); gq=abs(qf-best)/max(1,abs(best))
  if gl<=1e-4 and gq<=1e-4: verdict='tie'; c['tie']+=1
  elif lf<qf: verdict='LazyIPDCA'; c['LazyIPDCA']+=1
  else: verdict='QPALM'; c['QPALM']+=1
 out.append({'tag':tag,'n':nL,'m':mL,'lazy_objective':lf,'lazy_feas':lfe,'lazy_stat':l['stat'],'lazy_comp':l['comp'],'lazy_telemetry_kkt1e6':l['telemetry_kkt1e6'],'lazy_time_s':l['time_s'],'lazy_status':l['status'],'qpalm_objective':qf,'qpalm_feas':qfe,'qpalm_stat_rel':q['stat_rel'],'qpalm_comp_rel':q['comp_rel'],'qpalm_kkt1_ok':q['kkt1_ok'],'qpalm_time_s':q['wall_time_s'],'qpalm_status':q['status'],'dimension_aligned':int(aligned),'verdict':verdict})
fields=list(out[0].keys())
with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(out)
print('rows=',len(out),c); print('CSV =',a.out)
