"""Stage: specialist_probe. Tier: cpu. Trained domain-model baselines (R1-W1 "specialised models", AI review
"trained domain-model ceiling"): the standard OPERA-benchmark protocol (frozen encoder + linear probe) with the
cardiopulmonary foundation encoders OPERA-CT (Zhang et al. 2024) and HeAR (Baur et al. 2024), plus CLAP.

Per task, rotation round r (1-4) and report field: a class-balanced logistic regression on recording-level
features (mean of the recording's unit embeddings, standardised) is trained on exactly the AuscultRAG
knowledge base of that round (same corpus, all folds except r, patient-disjoint from the test fold) and
predicts the main-run test queries of fold r. C is chosen on dev only (train folds 1-4 -> predict fold 0,
mean report score over tasks). Scored with report_score.score_block (same informative-field report score).

Usage: python scripts/specialist_probe.py
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from auscultrag.llm.report_fields import FIELDS  # noqa: E402
from auscultrag.retrieval.store import label_table, load_embeddings, load_tasks  # noqa: E402
from report_score import score_block  # noqa: E402

warnings.filterwarnings("ignore")
C_GRID = [0.01, 0.1, 1.0]
OUT = ROOT / "artifacts_committed" / "final"


def rec_features(encoder: str) -> tuple[np.ndarray, dict]:
    emb, units = load_embeddings(encoder)
    idx = units.groupby("uid").indices
    uids = list(idx)
    X = np.vstack([emb[idx[u]].mean(axis=0) for u in uids])
    return X, {u: i for i, u in enumerate(uids)}


def fit_predict(Xtr, ytr, Xte, C):
    if len(set(ytr)) < 2:
        return [ytr[0]] * len(Xte)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    return list(clf.predict(sc.transform(Xte)))


def run(X, pos, tasks, labels, gt_map, queries_by_task, C, rounds):
    """Returns {task: rows} with gt / pred fields for the given query rounds."""
    out = defaultdict(list)
    for t, qs in queries_by_task.items():
        td = tasks[t]
        fields = FIELDS[t]
        recs = labels[(labels.corpus == td["corpus"]) & (labels.family == td["family"]) & labels.fold.notna()]
        recs = recs[[u in pos and (t, u) in gt_map for u in recs.index]]
        for r in rounds:
            q = [x for x in qs if x["round"] == r]
            if not q:
                continue
            tr = recs[recs.fold != r]
            Xtr = X[[pos[u] for u in tr.index]]
            Xte = X[[pos[x["uid"]] for x in q]]
            preds = {}
            for f in fields:
                y = [gt_map[(t, u)].get(f) for u in tr.index]
                keep = [i for i, v in enumerate(y) if v is not None]
                preds[f] = fit_predict(Xtr[keep], [y[i] for i in keep], Xte, C) if keep else [None] * len(q)
            for i, x in enumerate(q):
                out[t].append({"gt": x["gt_fields"], "pred": {f: preds[f][i] for f in fields}, "patient_id": x["patient_id"]})
    return out


def report(rows_by_task, fields_of):
    res = {}
    for t, rows in rows_by_task.items():
        fs = score_block(rows, fields_of[t])
        inf = [f for f, v in fs.items() if v["informative"]]
        prim = next(iter(fields_of[t]))
        res[t] = {"report_score": float(np.mean([fs[f]["macro_f1"] for f in inf])),
                  "primary_macro_f1": fs[prim]["macro_f1"], "primary_accuracy": fs[prim]["accuracy"], "n": len(rows)}
    return res


def main() -> None:
    tasks, labels = load_tasks(), label_table()
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {(t, u): dict(zip(g.field, g.value)) for (t, u), g in gt.groupby(["task", "uid"], sort=False)}
    test = [json.loads(l) for l in (ARTIFACT_ROOT / "llm" / "report_test_v3" / "queue.jsonl").open()]
    test_by_task = defaultdict(list)
    for x in test:
        if x["condition"] == "rag":
            test_by_task[x["task"]].append(x)
    fields_of = {t: {f: list(v[0]) for f, v in FIELDS[t].items()} for t in test_by_task}
    # dev queries: every fold-0 recording with a ground truth (one per patient)
    dev_by_task = defaultdict(list)
    for t in test_by_task:
        td = tasks[t]
        d = labels[(labels.corpus == td["corpus"]) & (labels.family == td["family"]) & (labels.fold == 0)]
        seen = set()
        for u, row in d.iterrows():
            if (t, u) in gt_map and row.patient_id not in seen:
                seen.add(row.patient_id)
                dev_by_task[t].append({"uid": u, "round": 0, "gt_fields": gt_map[(t, u)], "patient_id": row.patient_id})
    rows = []
    for enc in ("clap", "operaCT", "hear"):
        X, pos = rec_features(enc)
        dev_by_task_e = {t: [q for q in qs if q["uid"] in pos] for t, qs in dev_by_task.items()}
        dev = {C: np.mean([v["report_score"] for v in report(run(X, pos, tasks, labels, gt_map, dev_by_task_e, C, [0]),
                                                                  fields_of).values()]) for C in C_GRID}
        C = max(dev, key=dev.get)
        res = report(run(X, pos, tasks, labels, gt_map, test_by_task, C, [1, 2, 3, 4]), fields_of)
        for t, v in res.items():
            rows.append({"model": f"{enc}+linear-probe", "encoder": enc, "C_dev": C, "task": t, **v})
        print(enc, "dev C scores", {k: round(v, 3) for k, v in dev.items()}, "-> C", C, flush=True)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "specialist_probe.csv", index=False)
    w = d.pivot(index="task", columns="model", values="report_score")
    w.loc["mean"] = w.mean()
    w.round(3).to_csv(OUT / "specialist_probe_wide.csv")
    print(w.round(3).to_string())


if __name__ == "__main__":
    main()
