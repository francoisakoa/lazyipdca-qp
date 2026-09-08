#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/audit_public_repo.py
[ -d .git ] || git init
git branch -M main
git add .
if ! git diff --cached --quiet; then git commit -m "Initial reproducibility release"; fi
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/francoisakoa/ipdca-qp.git
echo "Ready. Review the commit, then run: git push -u origin main"
