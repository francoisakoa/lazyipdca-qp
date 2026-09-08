#!/usr/bin/env python3
from pathlib import Path
import csv
p=Path(__file__).resolve().parents[1]/'results/reference/numeric_manifest.csv'
with p.open(newline='') as f: rows=list(csv.DictReader(f))
for r in rows: print(f"{r['metric']:<45} {r['value']}")
