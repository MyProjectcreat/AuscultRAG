"""Stage: build_report_gt. Tier: cpu. Ground-truth report fields for every labelled recording of every task.

Output: $AUSCULTRAG_ARTIFACTS/reports/report_gt.parquet  (task, uid, field, value) -- value None = not annotated.
Fails loudly if any extracted value falls outside the field's allowed values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from auscultrag.llm.report_fields import FIELDS, ground_truth  # noqa: E402
from auscultrag.retrieval.store import label_table, load_tasks  # noqa: E402


def main() -> None:
    labels = label_table()
    rows, bad = [], []
    for task, t in load_tasks().items():
        sub = labels[(labels.corpus == t["corpus"]) & (labels.family == t["family"]) & labels[t["label_field"]].notna()]
        for uid, r in sub.iterrows():
            gt = ground_truth(task, r)
            for field, (allowed, _) in FIELDS[task].items():
                v = gt.get(field)
                if v is not None and v not in allowed:
                    bad.append((task, uid, field, v))
                rows.append({"task": task, "uid": uid, "field": field, "value": v})
    if bad:
        print(f"{len(bad)} values outside allowed sets, e.g. {bad[:10]}")
        sys.exit(1)
    df = pd.DataFrame(rows)
    out = ARTIFACT_ROOT / "reports" / "report_gt.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    pd.set_option("display.width", 200)
    for task, g in df.groupby("task"):
        print(f"== {task}")
        for field, gf in g.groupby("field", sort=False):
            vc = gf.value.fillna("<not annotated>").value_counts().to_dict()
            print(f"   {field:22s} {vc}")
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
