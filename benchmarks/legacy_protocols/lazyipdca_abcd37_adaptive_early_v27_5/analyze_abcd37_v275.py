#!/usr/bin/env python3
from pathlib import Path
import csv,json,math,re,sys
root=Path(sys.argv[1])
raw=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines() if x.strip()]
if len(raw)!=148:
    raise SystemExit(f'[FAIL] expected 148 records, found {len(raw)}')
R=[]
for rr in raw:
    j=json.load(open(rr['json']))
    txt=Path(rr['log']).read_text(errors='replace')
    def fv(*ks):
        for k in ks:
            if j.get(k) is not None:
                try:return float(j[k])
                except:pass
        return math.nan
    def sv(*ks):
        for k in ks:
            if j.get(k) is not None:return str(j[k])
        return '-'
    take=re.findall(r'\[LazyDCA-C-v275\] adaptive EARLY TAKEOVER it=(\d+) rank=(\d+)',txt)
    commits=re.findall(r'\[LazyDCA-C-v275\] progressive COMMIT it=(\d+) rank=(\d+)->(\d+)',txt)
    x=dict(arm=rr['arm'],tag=rr['tag'],rc=int(rr['rc']),wall=float(rr['wall_sec']),
           status=sv('status_str','status'),obj=fv('obj','objective'),
           feas=fv('kkt_feas_rel_original','kkt_feas_rel','cv'),
           it=int(fv('iters','iterations')),
           atoms=int(fv('n_atoms')) if math.isfinite(fv('n_atoms')) else -1,
           lanczos=int(fv('n_lanczos_calls')) if math.isfinite(fv('n_lanczos_calls')) else -1,
           take=len(take),take_it=int(take[0][0]) if take else -1,
           take_rank=int(take[0][1]) if take else -1,commits=len(commits))
    x['feasible']=x['feas']<=1e-6
    R.append(x)
D={(x['arm'],x['tag']):x for x in R}
tags=sorted({x['tag'] for x in R}); arms=('A','B','C','D')
for tag in tags:
    for arm in arms:
        if (arm,tag) not in D: raise SystemExit(f'[FAIL] missing {arm}/{tag}')
bad=[x for x in R if x['arm'] in ('A','B') and x['take']>0]
if bad:
    raise SystemExit('[FAIL] EARLY contaminated A/B: '+','.join(f"{x['arm']}/{x['tag']}" for x in bad))
TIE=1e-4
best_counts={a:0 for a in arms}; ratio_one={a:0 for a in arms}
rows=[]
for tag in tags:
    feasible=[D[(a,tag)] for a in arms if D[(a,tag)]['feasible']]
    fbest=min(x['obj'] for x in feasible) if feasible else math.nan
    successful=[]
    for a in arms:
        x=D[(a,tag)]
        gap=abs(x['obj']-fbest)/max(1.0,abs(fbest)) if x['feasible'] and math.isfinite(fbest) else math.inf
        if x['feasible'] and gap<=TIE:
            best_counts[a]+=1; successful.append(x)
    if successful:
        tmin=min(x['wall'] for x in successful)
        for x in successful:
            if abs(x['wall']-tmin)<=1e-12*max(1.0,tmin): ratio_one[x['arm']]+=1
    row={'tag':tag,'fbest':fbest}
    for a in arms:
        x=D[(a,tag)]
        gap=abs(x['obj']-fbest)/max(1.0,abs(fbest)) if x['feasible'] and math.isfinite(fbest) else math.inf
        row.update({f'{a}_obj':x['obj'],f'{a}_feas':x['feas'],f'{a}_wall':x['wall'],
                    f'{a}_status':x['status'],f'{a}_it':x['it'],f'{a}_atoms':x['atoms'],
                    f'{a}_lanczos':x['lanczos'],f'{a}_gap':gap,
                    f'{a}_best_tied':bool(x['feasible'] and gap<=TIE),
                    f'{a}_takeover':x['take'],f'{a}_commits':x['commits']})
    rows.append(row)
print('\n===== FULL 37 A/B/C/D — FEASIBILITY / OBJECTIVE / TIME =====')
print('arm  feasible  best/tied  geom_time(s)  ratio_one  total_wall(s)  iters  atoms  Lanczos  takeover/commit')
for a in arms:
    rr=[x for x in R if x['arm']==a]; times=[x['wall'] for x in rr]
    gm=math.exp(sum(math.log(max(t,1e-12)) for t in times)/len(times))
    print(f"{a:>2s}   {sum(x['feasible'] for x in rr):2d}/37     {best_counts[a]:2d}/37      {gm:10.3f}     {ratio_one[a]:2d}/37     {sum(times):12.3f}  {sum(x['it'] for x in rr):5d}  {sum(max(0,x['atoms']) for x in rr):5d}  {sum(max(0,x['lanczos']) for x in rr):7d}  {sum(x['take']>0 for x in rr):2d}/{sum(x['commits']>0 for x in rr):2d}")
print('\n===== PAIRWISE FEASIBLE OBJECTIVE COUNTS (relative tie 1e-4) =====')
for a,b in (('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')):
    lower=higher=tie=0
    for tag in tags:
        x=D[(a,tag)]; y=D[(b,tag)]
        if not (x['feasible'] and y['feasible']): continue
        scale=max(1.0,abs(x['obj']),abs(y['obj']))
        d=(x['obj']-y['obj'])/scale
        if d < -TIE: lower+=1
        elif d > TIE: higher+=1
        else: tie+=1
    print(f'{a} vs {b}: {lower} lower / {higher} higher / {tie} ties')
print('\n===== STATUS COUNTS =====')
for a in arms:
    c={}
    for x in R:
        if x['arm']==a:c[x['status']]=c.get(x['status'],0)+1
    print(f'{a}: {c}')
print('\n===== ADAPTIVE-ONLY EARLY CONTRACT =====')
for a in arms:
    rr=[x for x in R if x['arm']==a]
    print(f"{a}: takeover={sum(x['take']>0 for x in rr)}/37 commits={sum(x['commits']>0 for x in rr)}/37")
print('[PASS] A/B uncontaminated; C/D are the adaptive pair, with D=C+cap20.')
with (root/'abcd37_adaptive_early_v275.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"[INFO] CSV={root/'abcd37_adaptive_early_v275.csv'}")
