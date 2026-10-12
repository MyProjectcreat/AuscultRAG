"""Stage: select_retrieval_config. Tier: cpu. DEV ONLY (R4): picks one retrieval setting per task.

Rule (fixed before any LLM call): among dev-grid rows (round 0) with k in {5, 10}, take the highest kNN
macro-F1; settings within TIE of the best are broken towards simpler ones (smaller N, smaller m,
smaller k, centering on). Test rounds are never read.
Output: $AUSCULTRAG_ARTIFACTS/selection/retrieval_selection_dev.json (CLAP) or retrieval_selection_dev_<encoder>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402

TIE = 0.005
K_ALLOWED = (5, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="clap")
    enc = ap.parse_args().encoder
    d = ARTIFACT_ROOT / "retrieval" / enc
    df = pd.concat([pd.read_csv(p) for p in sorted(d.glob("summary_dev_grid_scope-corpus_cv-0_c*.csv"))])
    df = df[(df["round"] == 0) & df.k.isin(K_ALLOWED)]
    sel = {}
    for task, g in df.groupby("task"):
        best = g.macro_f1.max()
        cands = g[g.macro_f1 >= best - TIE].sort_values(["top_n", "m", "k", "center"], ascending=[True, True, True, False])
        r = cands.iloc[0]
        sel[task] = {"center": int(r.center), "top_n": int(r.top_n), "m": int(r.m), "k": int(r.k),
                     "retrieval_cfg": f"scope-corpus_cv-0_c{int(r.center)}_N{int(r.top_n)}_m{int(r.m)}_k20",
                     "dev_knn_macro_f1": round(float(r.macro_f1), 4), "dev_knn_accuracy": round(float(r.accuracy), 4),
                     "dev_knn_macro_f1_ci95": json.loads(r.macro_f1_ci95) if isinstance(r.macro_f1_ci95, str) else r.macro_f1_ci95,
                     "same_corpus_rate": round(float(r.same_corpus_rate), 4),
                     "label_precision_at_k": round(float(r.label_precision_at_k), 4),
                     "dev_queries": int(r.n_queries), "dev_patients": int(r.n_patients)}
    out = ARTIFACT_ROOT / "selection" / ("retrieval_selection_dev.json" if enc == "clap" else f"retrieval_selection_dev_{enc}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sel, indent=2))
    print(pd.DataFrame(sel).T[["center", "top_n", "m", "k", "dev_knn_macro_f1", "dev_knn_accuracy", "dev_queries"]].to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
