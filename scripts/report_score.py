"""Stage: report_score. Tier: cpu. Scores structured reports field by field against expert annotations.

For each task, condition (norag / rag) and field: macro-F1 and accuracy over queries whose ground truth is
annotated for that field; a missing / invalid field value counts as wrong. Report score = mean field
macro-F1 over INFORMATIVE fields, i.e. fields whose ground truth takes >= 2 values in the evaluated query set
(a field with a single observed class gives a trivial macro-F1 of 1.0 to anyone who says "no"); rule fixed
on dev before any test run. All fields are still listed in the per-field table. Baseline "knn_fieldwise": similarity-weighted vote of the retrieved cases' annotations, per field
(the same cases shown in the w/ RAG prompt). Patient-level bootstrap CIs on the report score.

Usage: python scripts/report_score.py --queue <dir>/queue.jsonl --model <model>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.eval.metrics import macro_f1  # noqa: E402

REPORT_RE = re.compile(r"REPORT\s*[:：]\s*(\{.*\})", re.IGNORECASE | re.DOTALL)


def parse_report(text: str | None, fields: dict[str, list[str]]) -> tuple[dict, bool]:
    """Returns ({field: value or None}, json_ok). Values are matched case-insensitively to allowed lists."""
    out = {f: None for f in fields}
    if not text:
        return out, False
    hits = REPORT_RE.findall(text)
    blob = hits[-1] if hits else None
    if blob is None:  # fall back to the last {...} block in the reply
        braces = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
        blob = braces[-1] if braces else None
    if blob is None:
        return out, False
    blob = blob[: blob.rfind("}") + 1]
    try:
        d = json.loads(blob)
    except json.JSONDecodeError:
        d = None
    if d is None:   # lenient fallback, same rule for every model: "key": "value" pairs after the last REPORT tag
        tail = re.split(r"REPORT\s*[:：]", text, flags=re.IGNORECASE)[-1] if re.search(r"REPORT\s*[:：]", text, re.I) else ""
        pairs = re.findall(r'"?([A-Za-z_ ]+)"?\s*:\s*"([^"]+)"', tail)
        if not pairs:
            return out, False
        d = {k.strip().lower().replace(" ", "_"): v for k, v in pairs}
        d = {f: d.get(f.lower()) for f in fields}
    for f, allowed in fields.items():
        v = d.get(f)
        if v is None:
            continue
        lut = {a.lower(): a for a in allowed}
        out[f] = lut.get(str(v).strip().lower())
    return out, True


def knn_fields(case_fields: list[dict], scores: list[float], fields: dict) -> dict:
    out = {}
    for f in fields:
        w = defaultdict(float)
        for cf, s in zip(case_fields, scores):
            if cf.get(f) is not None:
                w[cf[f]] += max(0.0, s)
        out[f] = max(w, key=w.get) if w else None
    return out


def score_block(rows: list[dict], fields: dict) -> dict:
    res = {}
    for f, allowed in fields.items():
        pairs = [(r["gt"][f], r["pred"][f] or "<invalid>") for r in rows if r["gt"].get(f) is not None]
        if not pairs:
            continue
        yt, yp = zip(*pairs)
        res[f] = {"n": len(pairs), "macro_f1": macro_f1(list(yt), list(yp), allowed),
                  "accuracy": float(np.mean(np.array(yt) == np.array(yp))), "informative": len(set(yt)) >= 2}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--knn-only", action="store_true", help="score only the kNN baselines (no responses needed)")
    a = ap.parse_args()
    items = [json.loads(l) for l in a.queue.open()]
    resp = {}
    rp = a.queue.with_name(f"responses_{a.model}.jsonl")
    for line in ([] if a.knn_only else rp.open()):
        r = json.loads(line)
        if r.get("error") is None or r["item_id"] not in resp:
            resp[r["item_id"]] = r
    per = defaultdict(list)          # (task, condition) -> rows
    for it in items:
        fields = it["fields"]
        pred, ok = parse_report(resp.get(it["item_id"], {}).get("text"), fields)
        if not a.knn_only:
            per[(it["task"], it["condition"])].append({"uid": it["uid"], "patient_id": it["patient_id"], "gt": it["gt_fields"],
                                                   "pred": pred, "json_ok": ok,
                                                   "complete": ok and all(v is not None for v in pred.values())})
        if it["condition"].startswith("rag"):     # kNN on the same evidence (controls: knn_rag_far, knn_rag_random)
            knn_cond = "knn_fieldwise" if it["condition"] == "rag" else f"knn_{it['condition']}"
            per[(it["task"], knn_cond)].append({"uid": it["uid"], "patient_id": it["patient_id"], "gt": it["gt_fields"],
                                                       "pred": knn_fields(it["case_fields"], it["case_scores"], fields),
                                                       "json_ok": True, "complete": True})
    fields_of = {it["task"]: it["fields"] for it in items}
    rng = np.random.default_rng(0)
    summary, field_rows = [], []
    for (task, cond), rows in sorted(per.items()):
        fields = fields_of[task]
        fs = score_block(rows, fields)
        informative = [f for f, v in fs.items() if v["informative"]]
        report = float(np.mean([fs[f]["macro_f1"] for f in informative]))
        pids = sorted({r["patient_id"] for r in rows})
        by_p = defaultdict(list)
        for r in rows:
            by_p[r["patient_id"]].append(r)
        boots = []
        for _ in range(a.n_boot):
            samp = [r for p in rng.choice(pids, size=len(pids)) for r in by_p[p]]
            b = score_block(samp, fields)
            vals = [b[f]["macro_f1"] for f in informative if f in b]
            boots.append(np.mean(vals) if vals else np.nan)
        summary.append({"task": task, "condition": cond, "n_queries": len(rows), "n_informative_fields": len(informative),
                        "report_score": report, "report_score_ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                        "json_ok_rate": float(np.mean([r["json_ok"] for r in rows])),
                        "complete_rate": float(np.mean([r["complete"] for r in rows]))})
        for f, v in fs.items():
            field_rows.append({"task": task, "condition": cond, "field": f, **v})
    s = pd.DataFrame(summary)
    fr = pd.DataFrame(field_rows)
    s.to_csv(a.queue.with_name(f"report_scores_{a.model}.csv"), index=False)
    fr.to_csv(a.queue.with_name(f"report_field_scores_{a.model}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(s.pivot(index="task", columns="condition", values="report_score").round(3).to_string())
    print("\njson_ok / complete rate:\n", s.pivot(index="task", columns="condition", values="complete_rate").round(2).to_string())
    print("\nper-field macro-F1:\n", fr.pivot_table(index=["task", "field"], columns="condition", values="macro_f1").round(3).to_string())


if __name__ == "__main__":
    main()
