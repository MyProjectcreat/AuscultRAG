"""Stage: run_retrieval. Tier: cpu.

For each task and rotation round, retrieves k diverse cases per query recording and scores the
non-LLM baseline (similarity-weighted kNN vote over the retrieved cases). Writes:
  $AUSCULTRAG_ARTIFACTS/retrieval/<encoder>/<config>/<task>/round<q>.jsonl   (cases per query, resumable)
  $AUSCULTRAG_ARTIFACTS/retrieval/<encoder>/<config>/summary.csv            (metrics + same-corpus rate)
Rounds: 0 = dev (tuning only), 1..4 = test. Never pass test rounds while choosing hyper-parameters.

Usage:
  python scripts/run_retrieval.py --rounds 0 --grid --center 1  # dev grid over N, m, k (repeat with --center 0)
  python scripts/run_retrieval.py --rounds 1,2,3,4 --top-n 200 --m 3 --k 10 --scope corpus
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, LABEL_SPACES, REPO_ROOT  # noqa: E402
from auscultrag.eval.metrics import summarise  # noqa: E402
from auscultrag.retrieval.engine import UnitIndex, retrieve_cases_multi  # noqa: E402
from auscultrag.retrieval.store import kb_mask, label_table, load_embeddings, load_tasks  # noqa: E402

GRID = {"top_n": [50, 200, 800], "m": [1, 3, 5], "k": [5, 10, 20]}


def case_label(labels: pd.DataFrame, uid: str, task: dict, tname: str, reg: dict):
    row = labels.loc[uid]
    grp = reg.get("task_group", {}).get(tname)
    if grp:
        field = reg["harmonized_groups"][grp].get(row["corpus"])
        return row[field] if field and pd.notna(row[field]) else None
    if row["corpus"] != task["corpus"]:
        return None
    v = row[task["label_field"]]
    return v if pd.notna(v) else None


def knn_vote(cases: list[dict], k: int) -> str | None:
    w = defaultdict(float)
    for c in cases[:k]:
        if c.get("label") is not None:
            w[c["label"]] += max(0.0, c["score"])
    return max(w, key=w.get) if w else None


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="clap")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--rounds", default="0")
    ap.add_argument("--scope", choices=["corpus", "family"], default="corpus")
    ap.add_argument("--coughvid", action="store_true", help="include COUGHVID KB padding")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--center", type=int, choices=[0, 1], default=1,
                    help="subtract the round's KB mean embedding and renormalise (label-free; counters CLAP anisotropy)")
    ap.add_argument("--grid", action="store_true", help="dev only: sweep GRID (k is swept by truncating k=max)")
    a = ap.parse_args()
    rounds = [int(r) for r in a.rounds.split(",")]
    if a.grid and rounds != [0]:
        sys.exit("--grid is for the dev round (0) only")

    reg = yaml.safe_load((REPO_ROOT / "configs" / "tasks.yaml").read_text())
    tasks = load_tasks()
    if a.tasks != "all":
        tasks = {t: tasks[t] for t in a.tasks.split(",")}
    emb, units = load_embeddings(a.encoder)
    labels = label_table()
    rec_units = units.groupby("uid").indices

    ns = GRID["top_n"] if a.grid else [a.top_n]
    ms = GRID["m"] if a.grid else [a.m]
    k_retrieve = max(GRID["k"]) if a.grid else a.k
    ks = GRID["k"] if a.grid else [a.k]
    combos = list(itertools.product(ns, ms))
    rows = []
    for (tname, t), q in itertools.product(tasks.items(), rounds):
        paths = {c: ARTIFACT_ROOT / "retrieval" / a.encoder /
                 f"scope-{a.scope}_cv-{int(a.coughvid)}_c{a.center}_N{c[0]}_m{c[1]}_k{k_retrieve}" / tname / f"round{q}.jsonl"
                 for c in combos}
        done = {c: {} for c in combos}
        for c, pth in paths.items():
            pth.parent.mkdir(parents=True, exist_ok=True)
            if pth.exists():
                for line in pth.open():
                    r = json.loads(line)
                    done[c][r["uid"]] = r
        qrecs = labels[(labels.corpus == t["corpus"]) & (labels.family == t["family"]) & (labels.fold == q)
                       & labels[t["label_field"]].notna()]
        todo = [u for u in qrecs.index if u in rec_units and any(u not in done[c] for c in combos)]
        if todo:
            mask = kb_mask(units, t["family"], q, t["corpus"], a.scope, a.coughvid)
            kb_emb = emb[mask]
            mu = kb_emb.mean(axis=0, keepdims=True) if a.center else None
            if a.center:
                kb_emb = _norm(kb_emb - mu)
            index = UnitIndex(kb_emb, units[mask])
            handles = {c: paths[c].open("a") for c in combos}
            try:
                for uid in todo:
                    qrow = labels.loc[uid]
                    query = {"uid": uid, "patient_id": qrow.patient_id, "dup_group_id": qrow.dup_group_id}
                    qe = emb[rec_units[uid]]
                    if a.center:
                        qe = _norm(qe - mu)
                    res = retrieve_cases_multi(index, qe, query, ns, ms, k_retrieve)
                    for c in combos:
                        if uid in done[c]:
                            continue
                        cases, diag = res[c]
                        for cs in cases:
                            cs["label"] = case_label(labels, cs["uid"], t, tname, reg)
                        rec = {"uid": uid, "patient_id": qrow.patient_id, "y_true": qrow[t["label_field"]],
                               "cases": cases, **diag}
                        handles[c].write(json.dumps(rec) + "\n")
                        done[c][uid] = rec
            finally:
                for h in handles.values():
                    h.close()
        space = list(LABEL_SPACES[t["label_space"]])
        for (top_n, m_) in combos:
            res = list(done[(top_n, m_)].values())
            for k in ks:
                df = pd.DataFrame({"patient_id": [r["patient_id"] for r in res], "y_true": [r["y_true"] for r in res],
                                   "y_pred": [knn_vote(r["cases"], k) for r in res]})
                same_corpus = np.mean([np.mean([c["corpus"] == t["corpus"] for c in r["cases"][:k]]) for r in res if r["cases"]])
                label_hit = np.mean([np.mean([c["label"] == r["y_true"] for c in r["cases"][:k]]) for r in res if r["cases"]])
                s = summarise(df, space, n_boot=300)
                rows.append({"task": tname, "round": q, "scope": a.scope, "coughvid": a.coughvid, "center": a.center,
                             "top_n": top_n, "m": m_, "k": k, **s, "same_corpus_rate": float(same_corpus),
                             "label_precision_at_k": float(label_hit)})
        best = max((r for r in rows if r["task"] == tname and r["round"] == q), key=lambda r: r["macro_f1"])
        print(f"{tname} r{q}: {len(qrecs)} queries; best knn macroF1={best['macro_f1']:.3f} "
              f"(N{best['top_n']} m{best['m']} k{best['k']})", flush=True)
    summ = pd.DataFrame(rows)
    tag = "dev_grid" if a.grid else f"rounds{a.rounds.replace(',', '')}_N{a.top_n}_m{a.m}_k{a.k}"
    path = ARTIFACT_ROOT / "retrieval" / a.encoder / f"summary_{tag}_scope-{a.scope}_cv-{int(a.coughvid)}_c{a.center}.csv"
    summ.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
