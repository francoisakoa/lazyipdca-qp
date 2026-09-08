#!/usr/bin/env python3
from pathlib import Path
import json, math, sys
root=Path(sys.argv[1])
raw=[json.loads(x) for x in (root/"runs.jsonl").read_text().splitlines() if x.strip()]
if len(raw)!=8:
    raise SystemExit(f"[FAIL] expected 8 preflight runs, found {len(raw)}")
def getv(j,*keys,default=None):
    for k in keys:
        if k in j and j[k] is not None:
            return j[k]
    return default
rows=[]; failures=[]
for rr in raw:
    j=json.load(open(rr["json"]))
    arm=rr["arm"]; tag=rr["tag"]
    mode=str(getv(j,"regularization_mode",default="")).lower()
    rho0=float(getv(j,"rho_initial",default=float("nan")))
    rhof=float(getv(j,"rho_final",default=float("nan")))
    upd=int(getv(j,"n_rho_model_updates",default=-1))
    inc=int(getv(j,"n_rho_model_increases",default=-1))
    dec=int(getv(j,"n_rho_model_decreases",default=-1))
    chk=int(getv(j,"n_rho_curvature_checks",default=-1))
    rej=int(getv(j,"n_rho_curvature_rejects",default=-1))
    fixed=bool(getv(j,"rho_model_fixed",default=False))
    low=bool(getv(j,"low_rank_effective",default=False))
    atoms=int(getv(j,"n_atoms",default=-1))
    feas=float(getv(j,"kkt_feas_rel_original","kkt_feas_rel","cv",default=float("nan")))
    obj=float(getv(j,"obj","objective",default=float("nan")))
    it=int(getv(j,"iters","iterations",default=-1))
    guard=getv(j,"rho_last_curvature_guard",default=float("nan"))
    try: guard=float(guard)
    except: guard=float("nan")
    ctrl=getv(j,"rho_controller_shared_dyn_lazy",default=None)
    relchg=abs(rhof-rho0)/max(1.0,abs(rho0))
    margin=(rhof+guard) if math.isfinite(guard) else float("nan")
    x=dict(arm=arm,tag=tag,mode=mode,rho0=rho0,rhof=rhof,upd=upd,inc=inc,dec=dec,chk=chk,rej=rej,
           fixed=fixed,low=low,atoms=atoms,feas=feas,obj=obj,it=it,guard=guard,margin=margin,relchg=relchg,shared=ctrl)
    rows.append(x)
    if not (math.isfinite(rho0) and math.isfinite(rhof) and rho0>=0 and rhof>=0):
        failures.append(f"{arm}/{tag}: rho non-finite or negative")
    if not math.isfinite(feas) or feas>1e-6:
        failures.append(f"{arm}/{tag}: infeasible return feas={feas}")
    if low:
        failures.append(f"{arm}/{tag}: low_rank_effective=true in scalar mode")
    if atoms not in (0,-1):
        failures.append(f"{arm}/{tag}: retained atoms={atoms} in scalar mode")
    if arm=="CONST":
        if mode!="const": failures.append(f"CONST/{tag}: regularization_mode={mode!r}")
        if not fixed: failures.append(f"CONST/{tag}: rho_model_fixed=false")
        if upd!=0 or inc!=0 or dec!=0:
            failures.append(f"CONST/{tag}: rho counters updates/inc/dec={upd}/{inc}/{dec}, expected 0/0/0")
        if relchg>1e-12:
            failures.append(f"CONST/{tag}: rho changed rho0={rho0} rhof={rhof} rel={relchg}")
    else:
        if mode!="dyn": failures.append(f"DYN/{tag}: regularization_mode={mode!r}")
        if fixed: failures.append(f"DYN/{tag}: rho_model_fixed=true")
        if upd>=0 and inc>=0 and dec>=0 and upd != inc+dec:
            failures.append(f"DYN/{tag}: inconsistent counters updates={upd} inc+dec={inc+dec}")
        if chk==0:
            failures.append(f"DYN/{tag}: zero curvature checks")
        if math.isfinite(margin):
            tol=1e-6*max(1.0,abs(rhof),abs(guard))
            if margin < -tol:
                failures.append(f"DYN/{tag}: final rho+last_guard={margin:.3e} violates curvature safety")
        if ctrl is False:
            failures.append(f"DYN/{tag}: shared DYN/LAZY rho controller reported false")
print("\n===== v2.7.5 RHO PREFLIGHT: CONST vs DYN =====")
print("arm    tag                       fixed lowrank atoms rho_initial      rho_final        relChange   upd inc dec chk rej  rho+guard      feas       it")
for x in rows:
    mg=f"{x['margin']:.3e}" if math.isfinite(x["margin"]) else "n/a"
    print(f"{x['arm']:6s} {x['tag']:25s} {int(x['fixed']):5d} {int(x['low']):7d} {x['atoms']:5d} "
          f"{x['rho0']:14.6e} {x['rhof']:14.6e} {x['relchg']:10.3e} "
          f"{x['upd']:4d} {x['inc']:3d} {x['dec']:3d} {x['chk']:3d} {x['rej']:3d} "
          f"{mg:>12s} {x['feas']:.3e} {x['it']:6d}")
dyn=[x for x in rows if x["arm"]=="DYN"]
dynamic_activity=sum(x["upd"]>0 for x in dyn)
dynamic_changed=sum(x["relchg"]>1e-12 for x in dyn)
if dynamic_activity==0:
    failures.append("DYN suite: no rho model update observed on any preflight case")
if dynamic_changed==0:
    failures.append("DYN suite: rho_final never differs from rho_initial")
print("\n===== PREFLIGHT SEMANTICS =====")
print(f"CONST fixed/invariant rows={sum(x['arm']=='CONST' and x['fixed'] and x['upd']==0 and x['relchg']<=1e-12 for x in rows)}/4")
print(f"DYN rows with model updates={dynamic_activity}/4; rows with rho change={dynamic_changed}/4")
marker=root/"RHO_PREFLIGHT_PASS"
if failures:
    print("\n[DECISION] RHO_PREFLIGHT_FAIL")
    for f in failures: print("[DIAG]",f)
    if marker.exists(): marker.unlink()
    raise SystemExit(2)
marker.write_text("PASS\n")
print("\n[DECISION] RHO_PREFLIGHT_PASS: CONST is fixed and DYN is behaviorally adaptive with scalar-only regularization.")
print(f"[PASS] marker={marker}")
