"""The vectorised retrieval must equal a plain reference implementation of the four operators."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.retrieval.engine import UnitIndex, retrieve_cases_multi  # noqa: E402


def reference(emb, units, q, query, n, m, k):
    sims = q @ emb.T
    pairs = []
    for qi in range(len(q)):
        top = np.argsort(-sims[qi], kind="stable")[:n]
        pairs += [(j, sims[qi, j]) for j in top]
    rows = [(units.uid[j], units.patient_id[j], s) for j, s in pairs
            if units.uid[j] != query["uid"] and units.patient_id[j] != query["patient_id"]
            and units.dup_group_id[j] != query["dup_group_id"]]
    df = pd.DataFrame(rows, columns=["rec", "pid", "sim"])
    score = df.groupby("rec").sim.apply(lambda s: s.sort_values(ascending=False).iloc[:m].mean())
    pid = df.groupby("rec").pid.first()
    out, seen = [], set()
    for rec in score.sort_values(ascending=False, kind="stable").index:
        if pid[rec] in seen:
            continue
        seen.add(pid[rec])
        out.append((rec, round(float(score[rec]), 5)))
        if len(out) == k:
            break
    return out


def test_vectorised_matches_reference():
    rng = np.random.default_rng(0)
    n_units = 400
    emb = rng.standard_normal((n_units, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rec = rng.integers(0, 80, n_units)
    units = pd.DataFrame({"uid": [f"r{r}" for r in rec], "patient_id": [f"p{r // 3}" for r in rec],
                          "dup_group_id": [f"d{r}" for r in rec], "corpus": "x"})
    index = UnitIndex(emb, units)
    q = emb[:5] + 0.01
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    query = {"uid": units.uid[0], "patient_id": units.patient_id[0], "dup_group_id": units.dup_group_id[0]}
    got = retrieve_cases_multi(index, q, query, [20, 60], [1, 3], 7)
    for n in (20, 60):
        for m in (1, 3):
            ref = reference(emb, units, q, query, n, m, 7)
            cases = got[(n, m)][0]
            assert [(c["uid"], c["score"]) for c in cases] == ref
            assert len({c["patient_id"] for c in cases}) == len(cases)
            assert all(c["patient_id"] != query["patient_id"] for c in cases)
