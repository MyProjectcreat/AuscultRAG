"""Case retrieval: unit-level nearest neighbours reduced to diverse, provenance-filtered cases.

The index side sees only (embedding, unit_id, recording uid, patient_id, dup_group_id, corpus).
Labels live in a separate table and are joined only after cases are selected (never used to rank).

For a query recording with units Q and a knowledge base of units K:
  1. top-N:      each query unit retrieves its N most cosine-similar KB units.
  2. provenance: drop candidates sharing the query's recording, patient or dup group.
  3. aggregate:  group surviving (query unit, KB unit) pairs by KB recording; score each recording
                 by the mean of its m strongest pair similarities.
  4. diversity:  greedily take the top-k recordings with pairwise-distinct patient_id.
All (N, m) settings are computed from ONE top-max(N) search (vectorised), so a dev grid costs one search.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetrievalParams:
    top_n: int = 200
    m: int = 3
    k: int = 10


def _codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, codes = np.unique(values, return_inverse=True)
    return uniq, codes


class UnitIndex:
    """Flat inner-product index over a subset of unit embeddings (unit-norm => cosine)."""

    def __init__(self, emb: np.ndarray, units: pd.DataFrame):
        assert len(emb) == len(units)
        units = units.reset_index(drop=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.ascontiguousarray(emb, dtype=np.float32))
        self.rec_uid, self.rec_code = _codes(units["uid"].to_numpy().astype(str))
        self.pid_vals, self.pid_code = _codes(units["patient_id"].to_numpy().astype(str))
        self.dup_vals, self.dup_code = _codes(units["dup_group_id"].to_numpy().astype(str))
        # per-recording attributes (every unit of a recording shares them)
        first = np.unique(self.rec_code, return_index=True)[1]
        self.rec_pid = self.pid_code[first]
        self.rec_corpus = units["corpus"].to_numpy().astype(str)[first]
        self._rec_lut = {u: i for i, u in enumerate(self.rec_uid)}
        self._pid_lut = {u: i for i, u in enumerate(self.pid_vals)}
        self._dup_lut = {u: i for i, u in enumerate(self.dup_vals)}

    def search(self, q: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(np.ascontiguousarray(q, dtype=np.float32), min(top_n, self.index.ntotal))


def retrieve_cases_multi(index: UnitIndex, q_emb: np.ndarray, query: dict, ns: list[int], ms: list[int],
                         k: int) -> dict[tuple[int, int], tuple[list[dict], dict]]:
    sims_all, idx_all = index.search(q_emb, max(ns))
    q_rec = index._rec_lut.get(query["uid"], -1)
    q_pid = index._pid_lut.get(query["patient_id"], -1)
    q_dup = index._dup_lut.get(query["dup_group_id"], -1)
    out = {}
    for n in ns:
        sims, idx = sims_all[:, :n].ravel(), idx_all[:, :n].ravel()
        ok = idx >= 0
        sims, idx = sims[ok], idx[ok]
        rec = index.rec_code[idx]
        keep = (rec != q_rec) & (index.pid_code[idx] != q_pid) & (index.dup_code[idx] != q_dup)
        n_filtered = int((~keep).sum())
        sims, rec = sims[keep], rec[keep]
        if len(rec) == 0:
            for m in ms:
                out[(n, m)] = ([], {"n_pairs": 0, "n_filtered": n_filtered, "n_candidate_recordings": 0})
            continue
        order = np.lexsort((-sims, rec))            # by recording, then similarity descending
        rec_s, sim_s = rec[order], sims[order]
        starts = np.r_[0, np.flatnonzero(np.diff(rec_s)) + 1]
        group_id = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rec_s)]))
        rank = np.arange(len(rec_s)) - starts[group_id]
        grp_rec = rec_s[starts]
        n_pairs = np.diff(np.r_[starts, len(rec_s)])
        for m in ms:
            sel = rank < m
            score = np.bincount(group_id[sel], weights=sim_s[sel], minlength=len(starts)) / np.minimum(n_pairs, m)
            by_score = np.argsort(-score, kind="stable")
            pids = index.rec_pid[grp_rec[by_score]]
            _, first_pos = np.unique(pids, return_index=True)   # first (best) recording per patient
            chosen = by_score[np.sort(first_pos)][:k]
            cases = [{"uid": str(index.rec_uid[grp_rec[g]]), "patient_id": str(index.pid_vals[index.rec_pid[grp_rec[g]]]),
                      "corpus": str(index.rec_corpus[grp_rec[g]]), "score": round(float(score[g]), 5),
                      "n_pairs": int(n_pairs[g])} for g in chosen]
            out[(n, m)] = (cases, {"n_pairs": int(len(rec)), "n_filtered": n_filtered,
                                   "n_candidate_recordings": int(len(starts))})
    return out


def retrieve_cases(index: UnitIndex, q_emb: np.ndarray, query: dict, p: RetrievalParams) -> tuple[list[dict], dict]:
    return retrieve_cases_multi(index, q_emb, query, [p.top_n], [p.m], p.k)[(p.top_n, p.m)]
