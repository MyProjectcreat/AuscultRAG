"""Stage: paired_diff. Tier: cpu. Paired comparison LLM vs kNN on the SAME queries and evidence.

For each (queue, model, LLM condition, kNN condition, task):
  * Delta = report_score(LLM) - report_score(kNN)  (report score = mean macro-F1 over informative fields)
  * paired patient-level bootstrap: each resample draws patients with replacement and recomputes BOTH scores on
    the same resample -> 95% CI of Delta and one-sided p = P(Delta* <= 0). Across tasks, the mean Delta uses a
    task-stratified bootstrap (patients resampled within each task).
  * field decomposition: Delta = sum_f Delta_f / F  (exact, F = informative fields), with a CI per field.
  * query decomposition (macro-F1 is not additive over queries, so we count): per field, queries where the
    LLM is right and kNN wrong (wins) vs LLM wrong and kNN right (losses), also split by true class.
Outputs (artifacts_committed/final/): paired_diff_tasks.csv, paired_diff_fields.csv, paired_diff_queries.csv,
paired_diff_pooled.csv.

Usage: python scripts/paired_diff.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from auscultrag.eval.metrics import macro_f1  # noqa: E402
from report_score import knn_fields, parse_report  # noqa: E402

LLM = ARTIFACT_ROOT / "llm"
OUT = ROOT / "artifacts_committed" / "final"
AUDIO = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3-flash-preview", "gpt-audio-1.5", "gpt-audio-2025-08-28",
         "gemini-3.5-flash-lite"]
COMPARISONS = ([("main", "report_test_v3", m, "rag", "knn_fieldwise") for m in AUDIO]
               + [("main_text", "report_test_v3_text", m, "text_rag", "knn_fieldwise") for m in ["gpt-5.4-2026-03-05", "gemini-3.5-flash"]]
               + [("xcorpus", "report_test_v3_xcorpus", m, c, f"knn_{c}") for m in ["gemini-3.5-flash", "gpt-audio-1.5"]
                  for c in ("rag@xc", "rag@in")])


def rows_for(run, model, llm_cond):
    items = [json.loads(l) for l in (LLM / run / "queue.jsonl").open()]
    resp = {}
    for l in (LLM / run / f"responses_{model}.jsonl").open():
        r = json.loads(l)
        if r.get("error") is None or r["item_id"] not in resp:
            resp[r["item_id"]] = r
    src = "rag" if llm_cond == "text_rag" else llm_cond       # kNN uses the evidence of the matching rag item
    out = defaultdict(list)
    for it in items:
        if it["condition"] != llm_cond:
            continue
        fields = it["fields"]
        llm, _ = parse_report(resp.get(it["item_id"], {}).get("text"), fields)
        knn = knn_fields(it["case_fields"], it["case_scores"], fields)
        out[it["task"]].append({"patient_id": it["patient_id"], "uid": it["uid"], "gt": it["gt_fields"],
                                "llm": llm, "knn": knn, "fields": fields})
    return out


def field_scores(rows, fields, informative, idx=None):
    rs = rows if idx is None else [rows[i] for i in idx]
    res = {}
    for f in informative:
        pairs = [(r["gt"][f], r["llm"][f] or "<invalid>", r["knn"][f] or "<invalid>") for r in rs if r["gt"].get(f) is not None]
        if not pairs:
            res[f] = (np.nan, np.nan)
            continue
        yt, yl, yk = zip(*pairs)
        res[f] = (macro_f1(list(yt), list(yl), fields[f]), macro_f1(list(yt), list(yk), fields[f]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(20261011)
    T, F, Q, P = [], [], [], []
    for exp, run, model, lc, kc in COMPARISONS:
        if not (LLM / run / f"responses_{model}.jsonl").exists():
            continue
        by_task = rows_for(run, model, lc)
        boot_task = {}
        for task, rows in sorted(by_task.items()):
            fields = rows[0]["fields"]
            informative = [f for f in fields if len({r["gt"].get(f) for r in rows if r["gt"].get(f) is not None}) >= 2]
            base = field_scores(rows, fields, informative)
            sl = np.mean([base[f][0] for f in informative])
            sk = np.mean([base[f][1] for f in informative])
            # paired patient-level bootstrap
            by_p = defaultdict(list)
            for i, r in enumerate(rows):
                by_p[r["patient_id"]].append(i)
            pids = list(by_p)
            bd, bf = [], defaultdict(list)
            for _ in range(a.n_boot):
                idx = [i for p in rng.choice(pids, size=len(pids)) for i in by_p[p]]
                fs = field_scores(rows, fields, informative, idx)
                d = [fs[f][0] - fs[f][1] for f in informative]
                bd.append(np.nanmean(d))
                for f, x in zip(informative, d):
                    bf[f].append(x)
            bd = np.array(bd)
            boot_task[task] = bd
            T.append({"experiment": exp, "model": model, "llm_condition": lc, "knn_condition": kc, "task": task,
                      "n_queries": len(rows), "n_informative_fields": len(informative), "llm_score": sl, "knn_score": sk,
                      "delta": sl - sk, "delta_ci95_low": np.percentile(bd, 2.5), "delta_ci95_high": np.percentile(bd, 97.5),
                      "p_one_sided_delta_le_0": float(np.mean(bd <= 0))})
            for f in informative:
                d = base[f][0] - base[f][1]
                v = np.array(bf[f])
                F.append({"experiment": exp, "model": model, "llm_condition": lc, "task": task, "field": f,
                          "llm_macro_f1": base[f][0], "knn_macro_f1": base[f][1], "delta_field": d,
                          "contribution_to_task_delta": d / len(informative),
                          "delta_field_ci95_low": np.nanpercentile(v, 2.5), "delta_field_ci95_high": np.nanpercentile(v, 97.5)})
                # query-level wins / losses, also by true class
                per_class = defaultdict(lambda: [0, 0, 0])
                for r in rows:
                    g = r["gt"].get(f)
                    if g is None:
                        continue
                    lo, ko = r["llm"][f] == g, r["knn"][f] == g
                    per_class[g][0] += 1
                    per_class[g][1] += int(lo and not ko)
                    per_class[g][2] += int(ko and not lo)
                for g, (n, w, l) in sorted(per_class.items()):
                    Q.append({"experiment": exp, "model": model, "llm_condition": lc, "task": task, "field": f,
                              "true_class": g, "n": n, "llm_right_knn_wrong": w, "knn_right_llm_wrong": l, "net": w - l})
        # pooled over tasks (task-stratified bootstrap: mean of per-task bootstrap draws)
        tasks = sorted(boot_task)
        pooled = np.mean([boot_task[t] for t in tasks], axis=0)
        sub = [r for r in T if r["experiment"] == exp and r["model"] == model and r["llm_condition"] == lc]
        P.append({"experiment": exp, "model": model, "llm_condition": lc, "n_tasks": len(tasks),
                  "mean_delta": float(np.mean([r["delta"] for r in sub])),
                  "ci95_low": np.percentile(pooled, 2.5), "ci95_high": np.percentile(pooled, 97.5),
                  "p_one_sided_delta_le_0": float(np.mean(pooled <= 0))})
        print(f"{exp:9s} {model:24s} {lc:9s} mean delta {P[-1]['mean_delta']:+.3f} "
              f"[{P[-1]['ci95_low']:+.3f}, {P[-1]['ci95_high']:+.3f}] p={P[-1]['p_one_sided_delta_le_0']:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(T).to_csv(OUT / "paired_diff_tasks.csv", index=False)
    pd.DataFrame(F).to_csv(OUT / "paired_diff_fields.csv", index=False)
    pd.DataFrame(Q).to_csv(OUT / "paired_diff_queries.csv", index=False)
    pd.DataFrame(P).to_csv(OUT / "paired_diff_pooled.csv", index=False)


if __name__ == "__main__":
    main()
