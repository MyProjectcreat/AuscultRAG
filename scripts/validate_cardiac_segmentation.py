"""Validation (cpu): cardiac segmenter vs CirCor expert S1/S2 annotations on a seeded sample.

Ground truth: CirCor .tsv rows (start, end, state) with state 1 = S1, 2 = systole, 3 = S2,
4 = diastole, 0 = unannotated. A detected unit boundary (S1 onset) is a hit if it lies within
+/- TOL of an annotated S1 onset; only boundaries inside the annotated span are scored.
Reports S1-onset precision/recall, units per recording, unit durations vs annotated cycle
durations, and a few concrete examples.

Usage: python scripts/validate_cardiac_segmentation.py [--n 60] [--tol 0.1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, SEED  # noqa: E402
from auscultrag.segmentation.base import segment_recording  # noqa: E402


def gt_s1_onsets(tsv: str) -> tuple[np.ndarray, float, float]:
    a = pd.read_csv(tsv, sep="\t", header=None, names=["start", "end", "state"])
    ann = a[a.state != 0]
    s1 = a.loc[a.state == 1, "start"].to_numpy()
    return s1, float(ann.start.min()) if len(ann) else 0.0, float(ann.end.max()) if len(ann) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tol", type=float, default=0.1)
    a = ap.parse_args()
    m = pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest_unsplit.parquet")
    sample = m[m.corpus == "circor"].sample(a.n, random_state=SEED)
    rows, examples = [], []
    for r in sample.itertuples():
        res = segment_recording(r.filepath, "cardiac")
        gt, lo, hi = gt_s1_onsets(r.meta_segmentation_tsv)
        det = np.array([u.start_sec for u in res.units] + ([res.units[-1].end_sec] if res.units else []))
        det_in = det[(det >= lo) & (det <= hi)]
        gt_in = gt[(gt >= lo) & (gt <= hi)]
        tp_det = sum(np.min(np.abs(gt_in - d)) <= a.tol for d in det_in) if len(gt_in) else 0
        tp_gt = sum(np.min(np.abs(det_in - g)) <= a.tol for g in gt_in) if len(det_in) else 0
        gt_cycles = np.diff(gt)
        rows.append({
            "uid": r.uid, "murmur": r.meta_murmur, "duration": r.duration_sec, "status": res.status,
            "n_units": len(res.units), "n_candidates": res.n_candidates,
            "period_est": res.diagnostics.get("period_sec"),
            "gt_median_cycle": float(np.median(gt_cycles)) if len(gt_cycles) else np.nan,
            "det_in_span": len(det_in), "gt_in_span": len(gt_in), "tp_det": tp_det, "tp_gt": tp_gt,
            "unit_durations": [round(u.duration_sec, 3) for u in res.units],
        })
        if len(examples) < 4:
            examples.append({"uid": r.uid, "units_first5": [(round(u.start_sec, 2), round(u.end_sec, 2)) for u in res.units[:5]],
                             "gt_s1_first6": [round(x, 2) for x in gt[:6]]})
    df = pd.DataFrame(rows)
    prec = df.tp_det.sum() / max(1, df.det_in_span.sum())
    rec = df.tp_gt.sum() / max(1, df.gt_in_span.sum())
    durs = np.concatenate([np.array(x) for x in df.unit_durations if len(x)]) if df.n_units.sum() else np.array([])
    per_rec_err = (df.period_est - df.gt_median_cycle).abs()
    summary = {
        "recordings": len(df), "tolerance_sec": a.tol,
        "status_counts": df.status.value_counts().to_dict(),
        "s1_onset_precision": round(float(prec), 3), "s1_onset_recall": round(float(rec), 3),
        "units_per_recording": df.n_units.describe().round(2).to_dict(),
        "unit_duration_sec": pd.Series(durs).describe(percentiles=[.05, .5, .95]).round(3).to_dict(),
        "period_abs_error_vs_gt_median_cycle_sec": per_rec_err.describe().round(3).to_dict(),
        "recall_by_murmur": df.groupby("murmur").apply(lambda g: round(g.tp_gt.sum() / max(1, g.gt_in_span.sum()), 3)).to_dict(),
        "examples": examples,
    }
    out = ARTIFACT_ROOT / "validation" / "cardiac_segmentation_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
