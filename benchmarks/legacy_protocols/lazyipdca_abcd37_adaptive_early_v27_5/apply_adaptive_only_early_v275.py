#!/usr/bin/env python3
import argparse, datetime, shutil
from pathlib import Path

MARK = "ADAPTIVE_ONLY_EARLY_TAKEOVER_V275"

def once(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {n}")
    return s.replace(old, new, 1)

def patch_text(s):
    if MARK in s:
        return s, False
    for t in ("EARLY_LOWRANK_TAKEOVER_V271", "guarded_progressive_takeover",
              "guarded_rank_now <= guarded_progressive_rank_gate",
              "st.atom_event_enable && !guarded_progressive_takeover",
              "EARLY low-rank TAKEOVER"):
        if t not in s:
            raise RuntimeError(f"not exact v2.7.1 lineage: missing {t}")

    old = (
        "const bool guarded_event_starved =\n"
        "    st.atom_event_enable && !guarded_progressive_takeover &&\n"
        "    event_feas_ok &&\n"
        "    guarded_rank_now <= guarded_progressive_rank_gate;\n"
    )
    new = (
        "const bool guarded_event_starved =\n"
        "    st.atom_event_enable && st.atom_oracle_enable &&\n"
        "    !guarded_progressive_takeover &&\n"
        "    event_feas_ok &&\n"
        "    guarded_rank_now <= guarded_progressive_rank_gate;\n"
    )
    s = once(s, old, new, "adaptive-only takeover gate")

    anchor = "  // EARLY_LOWRANK_TAKEOVER_V271\n"
    s = once(s, anchor, anchor +
        "  // ADAPTIVE_ONLY_EARLY_TAKEOVER_V275\n"
        "  // EARLY belongs only to adaptive selection; A/B keep legacy coverage behavior.\n",
        "v2.7.5 marker")

    s = s.replace("[LazyDCA-C-v271] EARLY low-rank TAKEOVER",
                  "[LazyDCA-C-v275] adaptive EARLY TAKEOVER")
    s = s.replace("[LazyDCA-C-v271] progressive COMMIT",
                  "[LazyDCA-C-v275] progressive COMMIT")
    s = s.replace("[LazyDCA-C-v271] progressive CAP",
                  "[LazyDCA-C-v275] progressive CAP")

    for t in (MARK, "st.atom_event_enable && st.atom_oracle_enable",
              "adaptive EARLY TAKEOVER", "[LazyDCA-C-v275] progressive COMMIT"):
        if t not in s:
            raise RuntimeError(f"postcondition missing {t}")
    return s, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = Path(a.root).resolve()/"src/solver.cpp"
    if not src.is_file():
        raise SystemExit(f"[FAIL] missing {src}")
    old = src.read_text(encoding="utf-8")
    try:
        new, changed = patch_text(old)
    except Exception as e:
        raise SystemExit(f"[FAIL] v2.7.5 patch refused: {e}")
    if not changed:
        print("[INFO] v2.7.5 already present"); return
    if a.dry_run:
        print("[PASS] v2.7.5 adaptive-only patch dry-run: contracts matched"); return
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    b=src.parent.parent/f".adaptive_only_early_v275_backup_{stamp}"
    b.mkdir(parents=True)
    shutil.copy2(src,b/"solver.cpp")
    src.write_text(new,encoding="utf-8")
    print(f"[PASS] v2.7.5 adaptive-only EARLY applied: {src}")
    print(f"[BACKUP] {b}")

if __name__ == "__main__":
    main()
