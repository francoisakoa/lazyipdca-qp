#!/usr/bin/env python3
import re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
terms=[
    'chat'+'gpt', 'open'+'ai', 'large language '+'model', 'generative '+'ai',
    'anthro'+'pic', 'clau'+'de', 'gem'+'ini', 'co'+'pilot'
]
patterns=[re.compile(re.escape(x),re.I) for x in terms]
patterns.append(re.compile(r'(?<![A-Za-z0-9_])' + ('l'+'lm') + r'(?![A-Za-z0-9_])',re.I))
errors=[]
for p in root.rglob('*'):
    if not p.is_file() or '.git' in p.parts:
        continue
    rel=str(p.relative_to(root))
    if any(rx.search(rel) for rx in patterns):
        errors.append(f'blocked token in path: {p}')
    if p.suffix.lower() in {'.png','.jpg','.jpeg','.pdf','.zip','.gz','.tar','.so','.a','.o','.pyc','.pyo'}:
        if p.suffix.lower() in {'.pyc','.pyo','.so','.a','.o'}:
            errors.append(f'build artifact: {p}')
        continue
    txt=p.read_text(errors='ignore')
    for rx in patterns:
        if rx.search(txt):
            errors.append(f'blocked terminology in {p}')
            break
    if '.before_' in p.name:
        errors.append(f'backup artifact: {p}')
if errors:
    print('\n'.join(errors))
    raise SystemExit(1)
print('[PASS] public repository audit')
