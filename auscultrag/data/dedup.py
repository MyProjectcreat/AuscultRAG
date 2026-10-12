"""Content-level cleaning applied after audio probing, across all corpora.

Rules (each exclusion/merge is counted in the build report):
  1. Digital silence (peak amplitude exactly 0) is excluded.
  2. Byte-identical audio filed under one patient in two different families (e.g. the same Coswara
     file uploaded as both "cough" and "breathing") has an unknowable modality: all copies excluded.
  3. Byte-identical audio under two or more patient_ids is evidence they are one person (or one
     upload duplicated): the patient_ids are merged (union-find) and one copy is kept.
  4. Remaining byte-identical copies within one patient and family: keep the first uid only.
After this, content_sha1 is unique across the manifest, which verify_invariants.py enforces.
"""
from __future__ import annotations

import pandas as pd

from auscultrag.corpora.common import BuildReport, UnionFind


def clean(m: pd.DataFrame, reports: dict[str, BuildReport]) -> pd.DataFrame:
    m = m.sort_values("uid").reset_index(drop=True)

    silent = m["peak_abs"].fillna(1.0) == 0.0
    for c, n in m.loc[silent, "corpus"].value_counts().items():
        reports[c].drop("digital silence (peak amplitude 0)", int(n))
    m = m.loc[~silent]

    groups = m.groupby("content_sha1")
    fam_conflict = groups["family"].transform("nunique") > 1
    for c, n in m.loc[fam_conflict, "corpus"].value_counts().items():
        reports[c].drop("identical audio filed under two families (modality unknown)", int(n))
    m = m.loc[~fam_conflict]

    uf = UnionFind()
    for _, sub in m.groupby("content_sha1"):
        pids = sorted(set(sub["patient_id"]))
        for p in pids[1:]:
            uf.union(pids[0], p)
    merged = m["patient_id"].map(uf.find)
    changed = merged != m["patient_id"]
    for c, sub in m.loc[changed].groupby("corpus"):
        targets = merged[sub.index]
        largest = targets.value_counts()
        reports[c].note(f"merged {sub['patient_id'].nunique()} patient_id(s) into {targets.nunique()} other(s) because "
                        f"they share byte-identical audio (largest cluster: {largest.index[0]} absorbs {largest.iloc[0]}); "
                        f"full mapping = manifest patient_id vs recording_id")
    m = m.assign(patient_id=merged.astype("string"))

    dup = m.duplicated("content_sha1", keep="first")
    for c, n in m.loc[dup, "corpus"].value_counts().items():
        reports[c].drop("byte-identical copy of another kept recording", int(n))
    return m.loc[~dup].reset_index(drop=True)
