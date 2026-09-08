#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--source',required=True)
ap.add_argument('--stage',required=True)
a=ap.parse_args()
src=Path(a.source).resolve(); dst=Path(a.stage).resolve()
if not (src/'src/solver.cpp').is_file():
    raise SystemExit(f'[FAIL] missing v2.7.1 source: {src}')
if dst.exists(): shutil.rmtree(dst)
def ignore(path,names):
    return [n for n in names if n.startswith(('build','logs','results','__pycache__')) or '_backup_' in n]
shutil.copytree(src,dst,ignore=ignore)
print(f'[PASS] copied v2.7.1 source to fresh v2.7.5 staging: {dst}')
