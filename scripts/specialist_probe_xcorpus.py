"""Stage: specialist_probe_xcorpus. Tier: cpu. Trained specialists under distribution shift (R2): the linear probe
of specialist_probe.py trained on the TWO OTHER lung corpora (harmonised fields adventitious / crackles /
wheezes, as in cross_corpus_prepare.py) and applied to the third, on the same queries and fields as the
rag@xc condition. C is the dev-selected value of specialist_probe.py for each encoder.

Usage: python scripts/specialist_probe_xcorpus.py
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from auscultrag.retrieval.store import label_table  # noqa: E402
from cross_corpus_prepare import TASKS, harm  # noqa: E402
from report_score import score_block  # noqa: E402
from specialist_probe import fit_predict, rec_features  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    sel_C = pd.read_csv(ROOT / "artifacts_committed" / "final" / "specialist_probe.csv").groupby("encoder").C_dev.first().to_dict()
    labels = label_table()
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {u: harm(dict(zip(g.field, g.value))) for (t, u), g in gt.groupby(["task", "uid"], sort=False) if t in TASKS}
    q = [json.loads(l) for l in (ARTIFACT_ROOT / "llm" / "report_test_v3_xcorpus" / "queue.jsonl").open()]
    q = [x for x in q if x["condition"] == "rag@xc"]
    rows = []
    for enc, C in sel_C.items():
        X, pos = rec_features(enc)
        for task, corpus in TASKS.items():
            qs = [x for x in q if x["task"] == task and x["uid"] in pos]
            fields = qs[0]["fields"]
            tr = [u for u in gt_map if labels.loc[u, "corpus"] != corpus and u in pos and pd.notna(labels.loc[u, "fold"])]
            Xtr, Xte = X[[pos[u] for u in tr]], X[[pos[x["uid"]] for x in qs]]
            preds = {}
            for f in fields:
                y = [gt_map[u].get(f) for u in tr]
                keep = [i for i, v in enumerate(y) if v is not None]
                preds[f] = fit_predict(Xtr[keep], [y[i] for i in keep], Xte, C)
            res = [{"gt": x["gt_fields"], "pred": {f: preds[f][i] for f in fields}} for i, x in enumerate(qs)]
            fs = score_block(res, fields)
            inf = [f for f, v in fs.items() if v["informative"]]
            rows.append({"model": f"{enc}+linear-probe@xc", "task": task, "n": len(qs),
                         "report_score": float(np.mean([fs[f]["macro_f1"] for f in inf]))})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "artifacts_committed" / "final" / "specialist_probe_xcorpus.csv", index=False)
    print(d.pivot(index="task", columns="model", values="report_score").round(3).to_string())


if __name__ == "__main__":
    main()
