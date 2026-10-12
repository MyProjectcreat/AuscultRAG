"""Patient-level folds (R2, R4) with a rotating test fold.

Each labelled corpus is cut into N_FOLDS patient-disjoint folds, stratified by patient label:
  * fold 0 is the DEV fold: all hyper-parameters (k, N, m, prompt variants) are selected with fold-0
    patients as queries and folds 1..N-1 as the knowledge base. It is never a test query set.
  * test rounds r = 1..N-1: queries = fold r, knowledge base = every other fold (including fold 0).
Every non-dev patient is a test query exactly once, so per-class test counts are ~4x what a single
60/20/20 split gives (TBScreen NTB: ~28 test patients instead of 7), and results come with
patient-level bootstrap intervals. Within any round, query and KB patients are disjoint.

The manifest's `split` column holds the round-1 view (fold 0 = dev, fold 1 = test, rest =
knowledge_base) so single-split tooling and the R14 verifier keep working; `fold` is authoritative.
Fold assignment: patient COUNT per (stratum, fold) fixed by largest-remainder rounding; which
patients go where is chosen among many seeded shuffles to keep each fold's RECORDING share near 1/N.
"""
from __future__ import annotations

import random

import pandas as pd

N_FOLDS = 5
DEV_FOLD = 0


def patient_table(manifest: pd.DataFrame) -> pd.DataFrame:
    g = manifest.groupby(["corpus", "patient_id"])
    labels = g["label"].agg(lambda s: "|".join(sorted({str(v) for v in s.dropna()})) or "<unlabelled>")
    return pd.DataFrame({"n_recordings": g.size(), "stratum": labels}).reset_index()


def fold_counts(n: int, n_folds: int = N_FOLDS) -> list[int]:
    """Patients per fold for a stratum of n. Remainders go to test folds (1..) before the dev fold, so
    a rare class reaches evaluation before it reaches tuning."""
    base, rem = divmod(n, n_folds)
    counts = [base] * n_folds
    order = list(range(1, n_folds)) + [DEV_FOLD]
    for f in order[:rem]:
        counts[f] += 1
    return counts


def assign_folds(patients: pd.DataFrame, seed: int, n_trials: int = 2000) -> pd.Series:
    rng = random.Random(seed)
    strata = {s: list(sub.index) for s, sub in patients.groupby("stratum", sort=True)}
    total = patients["n_recordings"].sum()
    best, best_cost = None, float("inf")
    for _ in range(n_trials):
        out = {}
        for idx in strata.values():
            idx = idx[:]
            rng.shuffle(idx)
            pos = 0
            for f, c in enumerate(fold_counts(len(idx))):
                for i in idx[pos : pos + c]:
                    out[i] = f
                pos += c
        a = pd.Series(out)
        recs = patients["n_recordings"].groupby(a).sum()
        cost = sum(abs(recs.get(f, 0) / total - 1 / N_FOLDS) for f in range(N_FOLDS))
        if cost < best_cost:
            best, best_cost = a, cost
    return best.reindex(patients.index)


def round_view(fold: int, test_fold: int) -> str:
    if fold == DEV_FOLD:
        return "dev"
    return "test" if fold == test_fold else "knowledge_base"


def split_manifest(manifest: pd.DataFrame, seed: int, policy: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = policy or {}
    pt = patient_table(manifest)
    pt["fold"] = pd.NA
    pt["split"] = pd.NA
    for corpus, sub in pt.groupby("corpus"):
        rule = policy.get(corpus, "folds")
        if rule == "folds":
            folds = assign_folds(sub, seed=seed + _stable_int(corpus))
            pt.loc[sub.index, "fold"] = folds
            pt.loc[sub.index, "split"] = folds.map(lambda f: round_view(int(f), 1))
        elif rule == "coughvid_expert_test":
            # Expert-annotated recordings: test-only OOD pool. The rest: optional KB padding (R7).
            expert = manifest[manifest.corpus == corpus].set_index("patient_id")["meta_expert_n"]
            n_exp = expert.groupby(level=0).apply(lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max()))
            has_exp = sub["patient_id"].map(lambda p: n_exp.get(p, 0) > 0)
            pt.loc[sub.index, "split"] = has_exp.map({True: "test", False: "knowledge_base"})
        else:
            raise ValueError(f"unknown split policy {rule}")
    key = pt.set_index(["corpus", "patient_id"])
    idx = pd.MultiIndex.from_arrays([manifest["corpus"], manifest["patient_id"]])
    out = manifest.copy()
    out["split"] = pd.array(key["split"].reindex(idx).to_numpy(), dtype="string")
    out["fold"] = pd.array(key["fold"].reindex(idx).to_numpy(), dtype="Int64")
    return out, pt


def _stable_int(s: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(s))
