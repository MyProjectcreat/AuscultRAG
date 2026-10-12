"""Stage: verify_invariants (R14). Tier: cpu. Exits non-zero if any invariant fails.

Checks (R14 as specified, plus closely related ones marked *):
  1. patient_id sets of knowledge_base / dev / test are pairwise disjoint (per corpus and globally)
  2. no recording_id appears in more than one split (per corpus)
  3. no dup_group_id is split across splits
  4. every has_closed_set_label row has a label inside its declared label space
  5. every manifest filepath exists and is readable
  6. no audio file is referenced by more than one uid
  7*. uid unique; every row has a split
  8*. every dup_group_id belongs to exactly one patient_id
  9*. byte-identical audio (content_sha1) never appears in two different splits, and is unique overall
 10*. R6: every labelled (corpus, family) has rows in all three splits
 11*. R7: COUGHVID rows never claim a closed-set label
 12*. folds: each patient / dup group / content / recording in one fold; every rotation round has
      disjoint query and KB patients and dup groups; `split` equals the round-1 view; COUGHVID rules
 13*. tasks (configs/tasks.yaml): labels inside the declared space; every task present in every fold

Usage: python scripts/verify_invariants.py [--manifest PATH]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, LABEL_SPACES, REPO_ROOT, SPLITS  # noqa: E402
from auscultrag.data.splits import DEV_FOLD, N_FOLDS  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ARTIFACT_ROOT / "manifest" / "manifest.parquet")
    ap.add_argument("--allow-missing-tasks", action="store_true",
                    help="tests only: skip registry tasks whose corpus is absent from the manifest")
    args = ap.parse_args()
    m = pd.read_parquet(args.manifest)
    print(f"manifest: {len(m)} rows, corpora: {m.corpus.value_counts().to_dict()}")

    check("uid unique", m.uid.is_unique, f"{m.uid.duplicated().sum()} duplicates")
    check("every row has a split in {knowledge_base,dev,test}", m.split.isin(SPLITS).all(),
          f"{(~m.split.isin(SPLITS)).sum()} rows without a valid split")

    # 1. patient disjointness
    pids = {s: set(m.loc[m.split == s, "patient_id"]) for s in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        inter = pids[a] & pids[b]
        check(f"patient_id disjoint: {a} vs {b}", not inter, f"|intersection| = {len(inter)}"
              + (f", e.g. {sorted(inter)[:3]}" if inter else ""))
    for corpus, sub in m.groupby("corpus"):
        n = sub.groupby("patient_id")["split"].nunique()
        check(f"[{corpus}] every patient in exactly one split", (n == 1).all(), f"{int((n > 1).sum())} patients span splits")

    # 2. recording_id
    n = m.groupby(["corpus", "recording_id"])["split"].nunique()
    check("no recording_id in more than one split", (n == 1).all(), f"{int((n > 1).sum())} offending recording_ids")

    # 3. dup groups
    n = m.groupby("dup_group_id")["split"].nunique()
    check("no dup_group_id split across splits", (n == 1).all(), f"{int((n > 1).sum())} offending groups")
    n = m.groupby("dup_group_id")["patient_id"].nunique()
    check("every dup_group_id within a single patient", (n == 1).all(), f"{int((n > 1).sum())} offending groups")

    # 4. labels inside their space
    closed = m[m.has_closed_set_label.fillna(False)]
    bad = [(r.uid, r.label, r.label_space_id) for r in closed.itertuples()
           if r.label_space_id not in LABEL_SPACES or r.label not in LABEL_SPACES[r.label_space_id]]
    check("closed-set labels inside declared label space", not bad, f"{len(bad)} bad rows" + (f", e.g. {bad[:3]}" if bad else ""))
    cv = m[m.corpus == "coughvid"]
    check("COUGHVID never has a closed-set label", (~cv.has_closed_set_label.fillna(False)).all() and cv.label.isna().all(),
          f"{len(cv)} COUGHVID rows")

    # 5. files exist and are readable
    missing = [p for p in m.filepath if not (os.path.isfile(p) and os.access(p, os.R_OK))]
    check("every filepath exists and is readable", not missing, f"{len(missing)} missing/unreadable"
          + (f", e.g. {missing[:2]}" if missing else ""))

    # 6. file referenced once; identical bytes never straddle splits
    real = m.filepath.map(os.path.realpath)
    check("no audio file referenced by more than one uid", real.is_unique, f"{real.duplicated().sum()} shared files")
    n = m.groupby("content_sha1")["split"].nunique()
    check("byte-identical audio never in two splits", (n == 1).all(), f"{int((n > 1).sum())} cross-split identical contents")
    check("content_sha1 unique (no byte-identical recordings survive cleaning)", m.content_sha1.is_unique,
          f"{int(m.content_sha1.duplicated().sum())} duplicate contents")

    # 10. R6
    for (corpus, family), sub in m[m.has_closed_set_label.fillna(False)].groupby(["corpus", "family"]):
        present = set(sub.split)
        check(f"[{corpus}/{family}] present in all three splits", present >= set(SPLITS), f"splits: {sorted(present)}")

    # 12*. Fold-level invariants for the rotating-test design (data/splits.py)
    lab = m[m.has_closed_set_label.fillna(False)]
    check("every labelled row has a fold in 0..N-1", lab.fold.notna().all() and lab.fold.between(0, N_FOLDS - 1).all(),
          f"{int(lab.fold.isna().sum())} rows without fold")
    for col in ("patient_id", "dup_group_id", "content_sha1", "recording_id"):
        n = lab.groupby(["corpus", col])["fold"].nunique()
        check(f"each {col} lies in exactly one fold", (n == 1).all(), f"{int((n > 1).sum())} offending")
    rounds = {"tuning (q=fold0)": DEV_FOLD, **{f"test round {r}": r for r in range(N_FOLDS) if r != DEV_FOLD}}
    for name, q in rounds.items():
        bad = 0
        for corpus, sub in lab.groupby("corpus"):
            kb = sub[sub.fold != q] if q != DEV_FOLD else sub[sub.fold != DEV_FOLD]
            qs = sub[sub.fold == q]
            bad += len(set(kb.patient_id) & set(qs.patient_id)) + len(set(kb.dup_group_id) & set(qs.dup_group_id))
        check(f"{name}: query/KB patient and dup-group intersection empty", bad == 0, f"|intersection| = {bad}")
    rv = lab.fold.map(lambda f: "dev" if f == DEV_FOLD else ("test" if f == 1 else "knowledge_base"))
    check("split column equals the round-1 view of fold", (rv == lab.split).all(), f"{int((rv != lab.split).sum())} mismatches")
    cvx = m[m.corpus == "coughvid"]
    check("COUGHVID: no fold, never dev, expert rows only in test",
          cvx.fold.isna().all() and not (cvx.split == "dev").any()
          and (cvx.loc[pd.to_numeric(cvx.meta_expert_n, errors="coerce").fillna(0) > 0, "split"] == "test").all()
          if len(cvx) else True)

    # 13*. Task registry: labels inside space, every task present in every fold
    tasks = yaml.safe_load((REPO_ROOT / "configs" / "tasks.yaml").read_text())["tasks"]
    for tname, t in tasks.items():
        sub = m[(m.corpus == t["corpus"]) & (m.family == t["family"])]
        if sub.empty:
            if args.allow_missing_tasks:
                print(f"[SKIP] task {tname}: corpus absent (--allow-missing-tasks)")
            else:
                check(f"task {tname}: has rows", False, "no rows")
            continue
        vals = sub[t["label_field"]].dropna()
        outside = set(vals) - set(LABEL_SPACES[t["label_space"]])
        check(f"task {tname}: labels inside {t['label_space']}", not outside, f"{len(vals)} labelled; outside: {sorted(outside)}")
        folds = set(sub.loc[sub[t["label_field"]].notna(), "fold"].dropna().astype(int))
        check(f"task {tname}: labelled rows in every fold", folds == set(range(N_FOLDS)), f"folds {sorted(folds)}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} invariant(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL INVARIANTS PASS")


if __name__ == "__main__":
    main()
