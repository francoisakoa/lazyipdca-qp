#!/usr/bin/env python3
import argparse,json,subprocess,time
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--binary',required=True); ap.add_argument('--arm',required=True)
ap.add_argument('--tag',required=True); ap.add_argument('--outdir',required=True)
ap.add_argument('--log',required=True); ap.add_argument('--record',required=True)
ap.add_argument('args',nargs=argparse.REMAINDER)
a=ap.parse_args()
args=list(a.args)
if args and args[0]=='--': args=args[1:]
out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
log=Path(a.log); log.parent.mkdir(parents=True,exist_ok=True)
t0=time.perf_counter()
with log.open('w') as fh:
    p=subprocess.run([a.binary]+args,stdout=fh,stderr=subprocess.STDOUT)
wall=time.perf_counter()-t0
js=list(out.rglob(f'{a.tag}_raw.json'))
if not js:
    tail='\n'.join(log.read_text(errors='replace').splitlines()[-20:])
    raise SystemExit(f'[FAIL] missing JSON {a.arm}/{a.tag} rc={p.returncode}\n{tail}')
rec={'arm':a.arm,'tag':a.tag,'rc':p.returncode,'wall_sec':wall,'json':str(js[0]),'log':str(log)}
Path(a.record).write_text(json.dumps(rec)+'\n')
print(f'[run] {a.arm:2s} {a.tag:24s} rc={p.returncode:2d} wall={wall:9.3f}s')
