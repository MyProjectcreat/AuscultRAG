"""Stage: llm_fusion. Tier: cpu. Late fusion of the LLM's label probabilities with the retrieval prior.

For every query:  p_fused = alpha * p_llm + (1 - alpha) * p_knn
  p_knn = similarity-weighted label distribution of the retrieved cases (the same cases shown in the prompt);
  p_llm = the model's "Probabilities:" line (renormalised); falls back to one-hot on its final answer,
          and to uniform when the reply is invalid.
alpha = 0 is the kNN baseline, alpha = 1 the LLM alone. Variants:
  * AuscultRAG      : p_llm from the w/ RAG reply (evidence in the prompt) fused with p_knn
  * late-fusion-only: p_llm from the w/o RAG reply fused with p_knn (no grounding)  -- ablation
ONE global alpha is chosen on DEV as the value maximising the mean macro-F1 over tasks (--select), and is
then frozen for test (--alpha). Usage:
  python scripts/llm_fusion.py --queue <dev queue> --select
  python scripts/llm_fusion.py --queue <test queue> --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auscultrag.eval.metrics import summarise  # noqa: E402
from llm_score import parse  # noqa: E402

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PROB_RE = re.compile(r"probabilities\s*[:：]\s*(.+)", re.IGNORECASE)


def parse_probs(text: str | None, labels: list[str]) -> np.ndarray | None:
    if not text:
        return None
    hits = PROB_RE.findall(text)
    if not hits:
        return None
    line = hits[-1]
    vals = []
    for lab in labels:
        m = re.search(re.escape(lab) + r"\s*[=:]\s*([0-9]*\.?[0-9]+)", line, re.IGNORECASE)
        if m is None:
            return None
        vals.append(float(m.group(1)))
    v = np.array(vals)
    return v / v.sum() if v.sum() > 0 else None


def llm_dist(text: str | None, labels: list[str]) -> tuple[np.ndarray, str]:
    p = parse_probs(text, labels)
    if p is not None:
        return p, "probabilities"
    ans = parse(text, labels)
    if ans is not None:
        return np.eye(len(labels))[labels.index(ans)], "answer_onehot"
    return np.full(len(labels), 1 / len(labels)), "uniform_invalid"


def knn_dist(case_labels: list, case_scores: list, labels: list[str]) -> np.ndarray:
    w = np.zeros(len(labels))
    for lab, sc in zip(case_labels, case_scores):
        if lab in labels:
            w[labels.index(lab)] += max(0.0, sc)
    return w / w.sum() if w.sum() > 0 else np.full(len(labels), 1 / len(labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--model", default="gpt-audio-2025-08-28")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true", help="DEV: sweep alpha and pick the best global value")
    g.add_argument("--alpha", type=float, help="TEST: apply a frozen alpha")
    a = ap.parse_args()

    items = [json.loads(l) for l in a.queue.open()]
    resp = {}
    for line in a.queue.with_name(f"responses_{a.model}.jsonl").open():
        r = json.loads(line)
        if r.get("error") is None or r["item_id"] not in resp:
            resp[r["item_id"]] = r
    by = {}
    for it in items:
        by.setdefault((it["task"], it["uid"]), {})[it["condition"]] = it
    recs, sources = [], []
    for (task, uid), d in by.items():
        if not {"rag", "norag"} <= set(d):
            continue
        labels = d["rag"]["labels"]
        pk = knn_dist(d["rag"]["case_labels"], d["rag"]["case_scores"], labels)
        pr, src_r = llm_dist(resp.get(d["rag"]["item_id"], {}).get("text"), labels)
        pn, src_n = llm_dist(resp.get(d["norag"]["item_id"], {}).get("text"), labels)
        sources += [(task, "rag", src_r), (task, "norag", src_n)]
        recs.append({"task": task, "uid": uid, "patient_id": d["rag"]["patient_id"], "y_true": d["rag"]["y_true"],
                     "labels": labels, "pk": pk, "pr": pr, "pn": pn})
    src = pd.DataFrame(sources, columns=["task", "condition", "source"])
    print("LLM distribution source:\n", src.groupby(["condition", "source"]).size().unstack(fill_value=0).to_string(), "\n")

    alphas = ALPHAS if a.select else [a.alpha]
    rows = []
    for task, group in pd.DataFrame(recs).groupby("task"):
        labels = group.labels.iloc[0]
        for variant, col in (("AuscultRAG", "pr"), ("late_fusion_only", "pn")):
            for al in alphas:
                pred = [labels[int(np.argmax(al * getattr(r, col) + (1 - al) * r.pk))] for r in group.itertuples()]
                df = pd.DataFrame({"patient_id": group.patient_id, "y_true": group.y_true, "y_pred": pred})
                s = summarise(df, labels, n_boot=500)
                rows.append({"task": task, "variant": variant, "alpha": al, **s})
    res = pd.DataFrame(rows)
    tag = "dev_alpha_sweep" if a.select else f"alpha{a.alpha}"
    res.to_csv(a.queue.with_name(f"fusion_{tag}_{a.model}.csv"), index=False)
    mean = res.groupby(["variant", "alpha"]).macro_f1.mean().unstack(0).round(4)
    print("mean macro-F1 over tasks (alpha=0 is kNN, alpha=1 is the LLM alone):\n", mean.to_string(), "\n")
    if a.select:
        best = float(mean["AuscultRAG"].idxmax())
        sel = {"alpha": best, "rule": "argmax over alpha of the mean dev macro-F1 across tasks (AuscultRAG variant)",
               "mean_dev_macro_f1": float(mean["AuscultRAG"].max())}
        a.queue.with_name("fusion_alpha_selected_dev.json").write_text(json.dumps(sel, indent=2))
        print("selected:", sel)
        piv = res[res.alpha.isin([0.0, best, 1.0])].pivot_table(index="task", columns=["variant", "alpha"], values="macro_f1").round(3)
        print(piv.to_string())


if __name__ == "__main__":
    main()
