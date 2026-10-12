"""Stage: analyze_evidence_use. Tier: cpu. How do the models use the retrieved evidence? (R1 minor, R2-W2, AI-W1)

Unit of analysis: a (query, field) pair, restricted to informative fields (as in report_score.py) with an
annotated ground truth. For each pair the retrieved neighbours give a similarity-weighted vote (= kNN
prediction) and an agreement level a = weight of the winning value / total weight.

  1. Stratified accuracy: a = 1 (unanimous) / 0.6 <= a < 1 (majority) / a < 0.6 (split).
     Hypothesis: when the neighbours conflict, a model that hears the audio can pick better than kNN or
     than the text-only model that sees the same evidence.
  2. Override analysis: how often the model's value differs from the kNN value, and who is right then.
  3. Out-of-evidence rate: model value not held by any neighbour (possible hallucination / own judgement).

Usage: python scripts/analyze_evidence_use.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from report_score import parse_report  # noqa: E402

LLM = ARTIFACT_ROOT / "llm"
OUT = Path(__file__).resolve().parents[1] / "artifacts_committed" / "analysis"
AUDIO_MODELS = ["gpt-audio-2025-08-28", "gpt-audio-1.5", "gemini-3-flash-preview", "gemini-3.5-flash",
                "gemini-3.5-flash-lite", "gemini-3.8-flash"]
TEXT_MODELS = ["gpt-5.4-2026-03-05", "gemini-3.5-flash"]


def load(queue: Path, model: str) -> tuple[list[dict], dict]:
    items = [json.loads(l) for l in queue.open()]
    resp = {}
    p = queue.with_name(f"responses_{model}.jsonl")
    if not p.exists():
        return items, {}
    for line in p.open():
        r = json.loads(line)
        if r.get("error") is None or r["item_id"] not in resp:
            resp[r["item_id"]] = r
    return items, resp


def vote(case_fields, scores, f):
    w = defaultdict(float)
    for cf, s in zip(case_fields, scores):
        if cf.get(f) is not None:
            w[cf[f]] += max(0.0, s)
    if not w or sum(w.values()) == 0:
        return None, 0.0, set()
    best = max(w, key=w.get)
    return best, w[best] / sum(w.values()), set(w)


def stratum(a: float) -> str:
    return "unanimous" if a >= 0.999 else ("majority" if a >= 0.6 else "split")


def informative_fields(items) -> dict[str, set]:
    vals = defaultdict(lambda: defaultdict(set))
    for it in items:
        for f, v in it["gt_fields"].items():
            if v is not None:
                vals[it["task"]][f].add(v)
    return {t: {f for f, s in fs.items() if len(s) >= 2} for t, fs in vals.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    runs = [(LLM / "report_test_v3" / "queue.jsonl", m, "rag", "audio+RAG") for m in AUDIO_MODELS] + \
           [(LLM / "report_test_v3_text" / "queue.jsonl", m, "text_rag", "text+RAG") for m in TEXT_MODELS]
    knn_done = False
    for queue, model, cond, label in runs:
        items, resp = load(queue, model)
        if not resp:
            print(f"skip {model} ({label}): no responses")
            continue
        inf = informative_fields(items)
        for it in items:
            if it["condition"] != cond:
                continue
            pred, _ = parse_report(resp.get(it["item_id"], {}).get("text"), it["fields"])
            for f in inf[it["task"]]:
                gt = it["gt_fields"].get(f)
                if gt is None:
                    continue
                kv, a, held = vote(it["case_fields"], it["case_scores"], f)
                base = {"task": it["task"], "uid": it["uid"], "field": f, "truth": gt, "knn": kv, "agreement": a,
                        "stratum": stratum(a), "gt_in_evidence": gt in held}
                rows.append({**base, "model": model, "setting": label, "pred": pred[f],
                             "out_of_evidence": pred[f] is not None and pred[f] not in held})
                if not knn_done and queue.parent.name == "report_test_v3":
                    rows.append({**base, "model": "kNN", "setting": "kNN", "pred": kv, "out_of_evidence": False})
        if queue.parent.name == "report_test_v3":
            knn_done = True
    d = pd.DataFrame(rows)
    d["correct"] = d.pred == d.truth
    d["knn_correct"] = d.knn == d.truth
    d["overrides"] = d.pred != d.knn
    d.to_csv(OUT / "evidence_use_pairs.csv.gz", index=False)

    # 1. accuracy by agreement stratum (pooled over tasks, and per task)
    strat = d.groupby(["setting", "model", "stratum"]).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                                           gt_in_evidence=("gt_in_evidence", "mean")).reset_index()
    strat.to_csv(OUT / "accuracy_by_agreement.csv", index=False)
    d.groupby(["task", "setting", "model", "stratum"]).correct.agg(["size", "mean"]).reset_index() \
        .rename(columns={"size": "n", "mean": "accuracy"}).to_csv(OUT / "accuracy_by_agreement_per_task.csv", index=False)

    # 2. overrides: model value != kNN value
    llm = d[d.setting != "kNN"]
    ov = llm.groupby(["setting", "model"]).apply(lambda g: pd.Series({
        "n_pairs": len(g), "override_rate": g.overrides.mean(),
        "acc_when_override": g[g.overrides].correct.mean(), "knn_acc_when_override": g[g.overrides].knn_correct.mean(),
        "net_gain_from_overrides": (g[g.overrides].correct.sum() - g[g.overrides].knn_correct.sum()) / len(g),
        "out_of_evidence_rate": g.out_of_evidence.mean(),
        "acc_out_of_evidence": g[g.out_of_evidence].correct.mean() if g.out_of_evidence.any() else np.nan,
    }), include_groups=False).reset_index()
    ov.to_csv(OUT / "override_analysis.csv", index=False)
    ov_split = llm[llm.stratum == "split"].groupby(["setting", "model"]).apply(lambda g: pd.Series({
        "n_pairs": len(g), "override_rate": g.overrides.mean(), "acc": g.correct.mean(), "knn_acc": g.knn_correct.mean(),
        "acc_when_override": g[g.overrides].correct.mean(), "knn_acc_when_override": g[g.overrides].knn_correct.mean(),
    }), include_groups=False).reset_index()
    ov_split.to_csv(OUT / "override_analysis_split.csv", index=False)

    pd.set_option("display.width", 220)
    print(strat.pivot_table(index=["setting", "model"], columns="stratum", values="accuracy").round(3)
          [["unanimous", "majority", "split"]].to_string())
    print("\nn per stratum:", strat.drop_duplicates("stratum").set_index("stratum").n.to_dict())
    print("\n", ov.round(3).to_string())
    print("\nsplit-evidence pairs only:\n", ov_split.round(3).to_string())


if __name__ == "__main__":
    main()
