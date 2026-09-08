#!/usr/bin/env python3
from pathlib import Path
import csv,json,math,sys
root=Path(sys.argv[1])
raw=[json.loads(x) for x in (root/"runs.jsonl").read_text().splitlines() if x.strip()]
if len(raw)!=111: raise SystemExit(f"[FAIL] expected 111 run records, found {len(raw)}")
R=[]
for rr in raw:
    j=json.load(open(rr["json"]))
    def fv(*ks):
        for k in ks:
            if j.get(k) is not None:
                try:return float(j[k])
                except:pass
        return math.nan
    def sv(*ks):
        for k in ks:
            if j.get(k) is not None:return str(j[k])
        return "-"
    x=dict(arm=rr["arm"],tag=rr["tag"],status=sv("status_str","status"),obj=fv("obj","objective"),
           feas=fv("kkt_feas_rel_original","kkt_feas_rel","cv"),it=int(fv("iters","iterations")),
           wall=float(rr["wall_sec"]),rho0=fv("rho_initial"),rhof=fv("rho_final"),
           rho_updates=int(fv("n_rho_model_updates")) if math.isfinite(fv("n_rho_model_updates")) else -1,
           atoms=int(fv("n_atoms")) if math.isfinite(fv("n_atoms")) else -1,
           lanczos=int(fv("n_lanczos_calls")) if math.isfinite(fv("n_lanczos_calls")) else -1,
           fac=int(fv("n_factorizations")) if math.isfinite(fv("n_factorizations")) else -1,
           mode=sv("regularization_mode"))
    x["feasible"]=math.isfinite(x["feas"]) and x["feas"]<=1e-6
    R.append(x)
D={(x["arm"],x["tag"]):x for x in R}; tags=sorted({x["tag"] for x in R}); arms=("CONST","DYN","LOWRANK")
rows=[]; wins={a:0 for a in arms}
print("\n===== v2.7.5 SCALAR REGULARIZATION AGAINST ADAPTIVE LOW-RANK METRIC =====")
print("tag                       CONST obj/t     DYN obj/t       LOWRANK obj/t    feasible C/D/L   best/tied")
for tag in tags:
    vals=[D[(a,tag)] for a in arms]
    feasible=[x for x in vals if x["feasible"] and math.isfinite(x["obj"])]
    fstar=min((x["obj"] for x in feasible),default=math.nan)
    tied=[]
    for x in vals:
        gap=abs(x["obj"]-fstar)/max(1.0,abs(fstar)) if x["feasible"] and math.isfinite(fstar) else math.inf
        x["gap"]=gap
        if gap<=1e-4: tied.append(x["arm"]); wins[x["arm"]]+=1
    c,d,l=vals
    print(f"{tag:26s} {c['obj']:+.6e}/{c['wall']:7.3f} {d['obj']:+.6e}/{d['wall']:7.3f} "
          f"{l['obj']:+.6e}/{l['wall']:7.3f}      {int(c['feasible'])}/{int(d['feasible'])}/{int(l['feasible'])}       {','.join(tied) if tied else '-'}")
    rows.append(dict(tag=tag,const_status=c["status"],dyn_status=d["status"],lowrank_status=l["status"],
        const_obj=c["obj"],dyn_obj=d["obj"],lowrank_obj=l["obj"],const_feas=c["feas"],dyn_feas=d["feas"],lowrank_feas=l["feas"],
        const_wall=c["wall"],dyn_wall=d["wall"],lowrank_wall=l["wall"],const_it=c["it"],dyn_it=d["it"],lowrank_it=l["it"],
        const_rho0=c["rho0"],const_rhof=c["rhof"],dyn_rho0=d["rho0"],dyn_rhof=d["rhof"],dyn_rho_updates=d["rho_updates"],
        lowrank_rho0=l["rho0"],lowrank_rhof=l["rhof"],lowrank_atoms=l["atoms"],
        const_lanczos=c["lanczos"],dyn_lanczos=d["lanczos"],lowrank_lanczos=l["lanczos"],
        const_fac=c["fac"],dyn_fac=d["fac"],lowrank_fac=l["fac"],best_tied=";".join(tied)))
print("\n===== AGGREGATES =====")
agg={}
for arm in arms:
    rr=[x for x in R if x["arm"]==arm]
    total=sum(x["wall"] for x in rr); feasible=sum(x["feasible"] for x in rr)
    geom=math.exp(sum(math.log(max(x["wall"],1e-12)) for x in rr)/len(rr))
    agg[arm]=(total,geom,feasible)
    print(f"{arm:8s} feasible={feasible}/37 best/tied={wins[arm]}/37 total_wall={total:.3f}s geom_wall={geom:.3f}s "
          f"iters={sum(x['it'] for x in rr)} atoms={sum(max(0,x['atoms']) for x in rr)} Lanczos={sum(max(0,x['lanczos']) for x in rr)} fac={sum(max(0,x['fac']) for x in rr)}")
lt=agg["LOWRANK"][0]
print("\n===== TIME RATIOS TO LOWRANK =====")
print(f"CONST / LOWRANK = {agg['CONST'][0]/max(lt,1e-300):.3f}")
print(f"DYN   / LOWRANK = {agg['DYN'][0]/max(lt,1e-300):.3f}")
print("Historical article reference only: fixed scalar 1372.29 s, adaptive scalar 1048.24 s.")
with (root/"scalar_vs_lowrank_37_v275.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
summary={"CONST":{"feasible":agg["CONST"][2],"best_tied":wins["CONST"],"total_wall":agg["CONST"][0],"geom_wall":agg["CONST"][1]},
         "DYN":{"feasible":agg["DYN"][2],"best_tied":wins["DYN"],"total_wall":agg["DYN"][0],"geom_wall":agg["DYN"][1]},
         "LOWRANK":{"feasible":agg["LOWRANK"][2],"best_tied":wins["LOWRANK"],"total_wall":agg["LOWRANK"][0],"geom_wall":agg["LOWRANK"][1]},
         "time_ratio_const_to_lowrank":agg["CONST"][0]/max(lt,1e-300),
         "time_ratio_dyn_to_lowrank":agg["DYN"][0]/max(lt,1e-300)}
(root/"scalar_vs_lowrank_summary_v275.json").write_text(json.dumps(summary,indent=2)+"\n")
print(f"[INFO] CSV={root/'scalar_vs_lowrank_37_v275.csv'}")
print(f"[INFO] SUMMARY={root/'scalar_vs_lowrank_summary_v275.json'}")
print("[PASS] same-binary 37-pair scalar-vs-low-rank campaign complete")
