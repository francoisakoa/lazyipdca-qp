#!/usr/bin/env python3
import csv,sys
with open(sys.argv[1],newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        name=(r.get('name') or r.get('problem') or '').strip(); n=(r.get('target_n') or '').strip()
        if name: print(f'{name}_n{n}' if n else name)
