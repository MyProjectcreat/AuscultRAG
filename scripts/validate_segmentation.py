"""Validation (cpu): run segment_row on a small seeded sample per (corpus, family) and report units
per recording, unit durations, drop rates and concrete examples. NOT a full-scale run.

Extra checks where a corpus provides a reference:
  * circor/cardiac: units come from expert annotations; the audio-only CardiacSegmenter is scored
    against them (S1-onset precision/recall within +/- 100 ms).
  * tbscreen/cough: padded clips (segment index -1) contain exactly one cough by construction, so the
    fraction of such clips yielding exactly one unit is a weak accuracy signal.

Usage: python scripts/validate_segmentation.py [--per-cell 25]
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
from auscultrag.segmentation.base import UNIT_BOUNDS_SEC, segment_recording, segment_row  # noqa: E402


def summarise(results: list[tuple[pd.Series, object]]) -> dict:
    n_units = [len(r.units) for _, r in results]
    n_cand = [r.n_candidates for _, r in results]
    durs = [u.duration_sec for _, r in results for u in r.units]
    rec_durs = [row["duration_sec"] for row, _ in results]
    return {
        "recordings": len(results),
        "recording_duration_sec": pd.Series(rec_durs).describe(percentiles=[.5]).round(2).to_dict(),
        "units_per_recording": pd.Series(n_units).describe(percentiles=[.5]).round(2).to_dict(),
        "recordings_with_zero_units": int(sum(x == 0 for x in n_units)),
        "candidates_dropped_by_plausibility": int(sum(n_cand) - sum(n_units)),
        "unit_duration_sec": pd.Series(durs).describe(percentiles=[.05, .5, .95]).round(3).to_dict() if durs else {},
        "examples": [{"uid": row["uid"], "dur": round(row["duration_sec"], 2),
                      "units": [(round(u.start_sec, 2), round(u.end_sec, 2)) for u in r.units[:6]]}
                     for row, r in results[:3]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=25)
    a = ap.parse_args()
    m = pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest_unsplit.parquet")
    report = {"unit_bounds_sec": UNIT_BOUNDS_SEC, "cells": {}}
    for (corpus, family), sub in m.groupby(["corpus", "family"]):
        if family == "respiratory" and corpus not in ("icbhi", "sprsound"):
            report["cells"][f"{corpus}/{family}"] = "no annotation; audio-only respiratory segmenter not implemented yet"
            continue
        sample = sub.sample(min(a.per_cell, len(sub)), random_state=SEED)
        results = [(row, segment_row(row)) for _, row in sample.iterrows()]
        report["cells"][f"{corpus}/{family}"] = summarise(results)
        print(f"done {corpus}/{family}", flush=True)

    # TBScreen weak check: padded single-cough clips should give exactly one unit.
    tb = m[(m.corpus == "tbscreen") & (pd.to_numeric(m.meta_segment_index, errors="coerce") == -1)].sample(200, random_state=SEED)
    counts = pd.Series([len(segment_row(row).units) for _, row in tb.iterrows()]).value_counts().sort_index()
    report["tbscreen_padded_clip_unit_counts"] = {int(k): int(v) for k, v in counts.items()}

    # Audio-only cardiac segmenter vs CirCor expert units.
    cc = m[m.corpus == "circor"].sample(60, random_state=SEED)
    tp_d = n_d = tp_g = n_g = 0
    for _, row in cc.iterrows():
        ref = np.array([u.start_sec for u in segment_row(row).units])
        det = np.array([u.start_sec for u in segment_recording(row["filepath"], "cardiac").units])
        if len(ref) == 0:
            continue
        lo, hi = ref.min() - 0.1, ref.max() + 0.1
        det = det[(det >= lo) & (det <= hi)]
        tp_d += sum(np.min(np.abs(ref - d)) <= 0.1 for d in det)
        tp_g += sum(np.min(np.abs(det - g)) <= 0.1 for g in ref) if len(det) else 0
        n_d, n_g = n_d + len(det), n_g + len(ref)
    report["audio_only_cardiac_vs_expert"] = {"recordings": 60, "tolerance_sec": 0.1,
                                              "precision": round(tp_d / max(1, n_d), 3),
                                              "recall": round(tp_g / max(1, n_g), 3)}
    out = ARTIFACT_ROOT / "validation" / "segmentation_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
