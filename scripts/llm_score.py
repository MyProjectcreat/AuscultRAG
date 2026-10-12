"""Stage: llm_score. Tier: cpu. Parses responses and scores both conditions on the SAME queries.

Parsing is strict and identical for both conditions: the last line matching "Final answer: <x>" is
taken and <x> must equal a valid label (case-insensitive, surrounding punctuation stripped). Anything
else is INVALID and counted as wrong; invalid rates are reported per condition. No keyword guessing.

Also reports, for the w/ RAG condition, how often the model's answer equals the majority label of the
retrieved cases (evidence-copy rate) and accuracy when the evidence majority is right vs wrong.
Usage: python scripts/llm_score.py --queue <dir>/queue.jsonl --model gpt-audio-2025-08-28
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.eval.metrics import summarise  # noqa: E402

FINAL_RE = re.compile(r"final answer\s*[:：]\s*(.+)", re.IGNORECASE)


def parse(text: str | None, labels: list[str]) -> str | None:
    if not text:
        return None
    hits = FINAL_RE.findall(text)
    if not hits:
        return None
    cand = hits[-1].strip().strip("*`'\".<>[]() ").strip()
    lut = {l.lower(): l for l in labels}
    return lut.get(cand.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--model", default="gpt-audio-2025-08-28")
    a = ap.parse_args()
    items = {json.loads(l)["item_id"]: json.loads(l) for l in a.queue.open()}
    resp = {}
    for line in a.queue.with_name(f"responses_{a.model}.jsonl").open():
        r = json.loads(line)
        if r.get("error") is None or r["item_id"] not in resp:
            resp[r["item_id"]] = r
    rows = []
    for iid, it in items.items():
        r = resp.get(iid, {})
        votes = Counter(l for l in it["case_labels"] if l is not None)
        rows.append({**{k: it[k] for k in ("task", "condition", "uid", "patient_id", "y_true")},
                     "y_pred": parse(r.get("text"), it["labels"]), "answered": r.get("text") is not None,
                     "evidence_majority": votes.most_common(1)[0][0] if votes else None, "labels": it["labels"]})
    # kNN baseline on exactly the same queries and the same retrieved cases (similarity-weighted vote).
    for iid, it in items.items():
        if it["condition"] != "rag":
            continue
        w = Counter()
        for lab, sc in zip(it["case_labels"], it.get("case_scores", [1.0] * len(it["case_labels"]))):
            if lab is not None:
                w[lab] += max(0.0, sc)
        rows.append({**{k: it[k] for k in ("task", "uid", "patient_id", "y_true")}, "condition": "knn",
                     "y_pred": w.most_common(1)[0][0] if w else None, "answered": True,
                     "evidence_majority": None, "labels": it["labels"]})
    df = pd.DataFrame(rows)
    df["y_pred"] = df.y_pred.fillna("<invalid>")
    out = []
    for (task, cond), g in df.groupby(["task", "condition"]):
        s = summarise(g, g.labels.iloc[0], n_boot=1000)
        extra = {}
        if cond == "rag":
            extra["copies_evidence_majority"] = float((g.y_pred == g.evidence_majority).mean())
            right = g.evidence_majority == g.y_true
            extra["acc_when_evidence_right"] = float((g[right].y_pred == g[right].y_true).mean()) if right.any() else None
            extra["acc_when_evidence_wrong"] = float((g[~right].y_pred == g[~right].y_true).mean()) if (~right).any() else None
            extra["evidence_majority_acc"] = float(right.mean())
        out.append({"task": task, "condition": cond, "unanswered": int((~g.answered).sum()), **s, **extra})
    res = pd.DataFrame(out)
    path = a.queue.with_name(f"scores_{a.model}.csv")
    res.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    cols = ["task", "condition", "n_queries", "accuracy", "macro_f1", "macro_f1_ci95", "invalid_rate",
            "copies_evidence_majority", "evidence_majority_acc"]
    print(res[[c for c in cols if c in res]].to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
