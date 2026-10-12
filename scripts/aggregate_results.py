"""Stage: aggregate_results. Tier: cpu. Collects every test-set result into paper tables (long CSVs + wide views)
under artifacts_committed/final/. Re-scores each (queue, model) with report_score.py so all numbers come from
one scorer. Missing runs are skipped (listed in final/README_status.txt).

  main.csv         audio-LLMs (w/o, w/ RAG), text-only controls, open-weight / domain audio-LLMs, kNN
  controls.csv     evidence relevance: rag vs rag_far vs rag_random (+ kNN on the same evidence)
  xcorpus.csv      distribution shift: norag / rag@in / rag@xc (+ kNN)
  textret.csv      text-retrieval baseline: query described in words -> retrieval (t2a / t2t)
  robustness.csv   query perturbations x {norag, rag, kNN} (LLM subset: 50 queries/task, 7 perturbations)
  robustness_knn.csv  kNN on all 785 robustness queries x 19 perturbations
  encoders.csv     encoder ablation: CLAP vs OPERA-CT vs HeAR (primary-label kNN and report-score kNN)

Usage: python scripts/aggregate_results.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from report_score import knn_fields, score_block  # noqa: E402

LLM = ARTIFACT_ROOT / "llm"
OUT = ROOT / "artifacts_committed" / "final"
RUNS = {
    "main": [("report_test_v3", m) for m in ["gpt-audio-2025-08-28", "gpt-audio-1.5", "gemini-3-flash-preview",
                                             "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash",
                                             "af3", "qwen2-audio-7b"]]
            + [("report_test_v3_text", m) for m in ["gpt-5.4-2026-03-05", "gemini-3.5-flash", "medgemma-4b", "medgemma-27b"]]
            + [(f"report_test_v3_extract_{m}", "qwen2.5-7b", f"{m}+extract") for m in ["af3", "qwen2-audio-7b"]],
    "controls": [("report_test_v3_controls", m) for m in ["gpt-audio-1.5", "gemini-3.5-flash"]],
    "xcorpus": [("report_test_v3_xcorpus", m) for m in ["gpt-audio-1.5", "gemini-3.5-flash"]],
    "textret": [("report_test_v3_textret", m) for m in ["gpt-audio-1.5"]],
    "robustness": [("report_test_v3_robust_llm", m) for m in ["gpt-audio-1.5"]],
}
MIN_COMPLETE = 0.98


def score(run: str, model: str, label: str | None = None) -> pd.DataFrame | None:
    q, r = LLM / run / "queue.jsonl", LLM / run / f"responses_{model}.jsonl"
    if not (q.exists() and r.exists()):
        return None
    n_q = sum(1 for _ in q.open())
    n_ok = len({json.loads(l)["item_id"] for l in r.open() if json.loads(l).get("error") is None})
    subprocess.run([sys.executable, str(ROOT / "scripts" / "report_score.py"), "--queue", str(q), "--model", model,
                    "--n-boot", "300"], check=True, capture_output=True)
    d = pd.read_csv(LLM / run / f"report_scores_{model}.csv")
    d["model"], d["run"], d["coverage"] = label or model, run, n_ok / n_q
    d.loc[d.condition.str.startswith("knn"), "model"] = "kNN"
    return d


def encoder_knn() -> pd.DataFrame:
    """Report-score kNN per encoder on the SAME test queries as the main run (each encoder with its own dev setting)."""
    items = [json.loads(l) for l in (LLM / "report_test_v3" / "queue.jsonl").open()]
    items = [it for it in items if it["condition"] == "rag"]
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {(t, u): dict(zip(g.field, g.value)) for (t, u), g in gt.groupby(["task", "uid"], sort=False)}
    rows = []
    for enc in ("clap", "operaCT", "hear"):
        sp = ARTIFACT_ROOT / "selection" / ("retrieval_selection_dev.json" if enc == "clap" else f"retrieval_selection_dev_{enc}.json")
        if not sp.exists():
            continue
        sel = json.loads(sp.read_text())
        cache = {}
        per = defaultdict(list)
        for it in items:
            t, s = it["task"], sel[it["task"]]
            key = (t, it["round"])
            if key not in cache:
                p = ARTIFACT_ROOT / "retrieval" / enc / s["retrieval_cfg"] / t / f"round{it['round']}.jsonl"
                cache[key] = {json.loads(l)["uid"]: json.loads(l) for l in p.open()} if p.exists() else {}
            rec = cache[key].get(it["uid"])
            if rec is None:
                continue
            cases = rec["cases"][: s["k"]]
            cf = [gt_map.get((t, c["uid"]), {}) for c in cases]
            per[t].append({"gt": it["gt_fields"], "pred": knn_fields(cf, [c["score"] for c in cases], it["fields"]),
                           "primary_ok": knn_fields(cf, [c["score"] for c in cases], it["fields"]).get(next(iter(it["fields"])))
                           == it["gt_fields"].get(next(iter(it["fields"])))})
        fields_of = {it["task"]: it["fields"] for it in items}
        for t, rs in per.items():
            fs = score_block(rs, fields_of[t])
            inf = [f for f, v in fs.items() if v["informative"]]
            prim = next(iter(fields_of[t]))
            rows.append({"encoder": enc, "task": t, "n_queries": len(rs),
                         "report_score_knn": float(np.mean([fs[f]["macro_f1"] for f in inf])),
                         "primary_macro_f1_knn": fs[prim]["macro_f1"], "primary_accuracy_knn": fs[prim]["accuracy"],
                         "dev_selection": json.dumps({k: sel[t][k] for k in ("center", "top_n", "m", "k")})})
    return pd.DataFrame(rows)


def supported_field_table() -> pd.DataFrame:
    """CaReAQA answers only some fields; compare every model on exactly those fields (same queries)."""
    from careaqa_run import SUPPORTED
    from report_score import parse_report
    cq = LLM / "report_test_v3_careaqa" / "queue.jsonl"
    if not cq.exists():
        return pd.DataFrame()
    uids = {(json.loads(l)["task"], json.loads(l)["uid"]) for l in cq.open()}
    items = [json.loads(l) for l in (LLM / "report_test_v3" / "queue.jsonl").open()]
    items = [it for it in items if (it["task"], it["uid"]) in uids]
    srcs = [("careaqa", LLM / "report_test_v3_careaqa" / "responses_careaqa.jsonl", "norag")]
    srcs += [(m, LLM / "report_test_v3" / f"responses_{m}.jsonl", c) for m in ("gpt-audio-1.5", "gemini-3.5-flash")
             for c in ("norag", "rag")]
    rows = []
    for name, path, cond in srcs + [("kNN", None, "rag")]:
        resp = {}
        if path is not None:
            for l in path.open():
                r = json.loads(l)
                if r.get("error") is None:
                    resp[r["item_id"]] = r["text"]
        per = defaultdict(list)
        for it in items:
            if it["condition"] != cond:
                continue
            fields = {f: it["fields"][f] for f in SUPPORTED[it["task"]]}
            if name == "kNN":
                pred = knn_fields(it["case_fields"], it["case_scores"], fields)
            else:
                pred, _ = parse_report(resp.get(it["item_id"]), fields)
            per[it["task"]].append({"gt": it["gt_fields"], "pred": pred})
        for t, rs in per.items():
            fields = {f: next(i for i in items if i["task"] == t)["fields"][f] for f in SUPPORTED[t]}
            fs = score_block(rs, fields)
            inf = [f for f, v in fs.items() if v["informative"]]
            rows.append({"model": name, "condition": cond, "task": t, "fields": ",".join(inf),
                         "supported_field_score": float(np.mean([fs[f]["macro_f1"] for f in inf])), "n": len(rs)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = []
    for table, runs in RUNS.items():
        parts = []
        for spec in runs:
            run, model = spec[0], spec[1]
            d = score(*spec)
            if d is None:
                status.append(f"{table}: {run}/{model} MISSING")
                continue
            cov = d.coverage.iloc[0]
            status.append(f"{table}: {run}/{model} coverage {cov:.3f}" + ("" if cov >= MIN_COMPLETE else "  (PARTIAL)"))
            parts.append(d)
        if not parts:
            continue
        d = pd.concat(parts)
        d = d.drop_duplicates(["run", "model", "condition", "task"])
        d.to_csv(OUT / f"{table}.csv", index=False)
        w = d.pivot_table(index=["run", "model", "condition"], columns="task", values="report_score").round(3)
        w["mean"] = w.mean(axis=1).round(3)
        w.to_csv(OUT / f"{table}_wide.csv")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "report_score.py"), "--queue",
                    str(LLM / "report_test_v3_robust" / "queue.jsonl"), "--model", "knn", "--knn-only", "--n-boot", "300"],
                   check=True, capture_output=True)
    rk = pd.read_csv(LLM / "report_test_v3_robust" / "report_scores_knn.csv")
    rk["perturbation"] = rk.condition.str.split("@").str[1]
    rk.to_csv(OUT / "robustness_knn.csv", index=False)
    sf = supported_field_table()
    if len(sf):
        sf.to_csv(OUT / "careaqa_supported_fields.csv", index=False)
        w = sf.pivot_table(index=["model", "condition"], columns="task", values="supported_field_score").round(3)
        w["mean"] = w.mean(axis=1).round(3)
        w.to_csv(OUT / "careaqa_supported_fields_wide.csv")
    e = encoder_knn()
    if len(e):
        e.to_csv(OUT / "encoders.csv", index=False)
        e.pivot(index="task", columns="encoder", values="report_score_knn").round(3).to_csv(OUT / "encoders_wide.csv")
    (OUT / "README_status.txt").write_text("\n".join(status) + "\n")
    print("\n".join(status))


if __name__ == "__main__":
    main()
