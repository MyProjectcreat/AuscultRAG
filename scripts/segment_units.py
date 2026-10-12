"""Stage: segment_units. Tier: cpu.

Segments every labelled recording (plus COUGHVID expert-annotated OOD recordings) into auscultation
units via auscultrag.segmentation.base.segment_row and writes one unit table.

Per recording at most MAX_UNITS units are kept, chosen evenly across the recording (deterministic),
so long CirCor/ICBHI files do not dominate the index. Recordings yielding no unit fall back to ONE
unit covering the whole recording (clipped to the family maximum) and are flagged, so every query
recording remains evaluable.

Output: $AUSCULTRAG_ARTIFACTS/units/units.parquet  (unit_id, uid, start_sec, end_sec, unit_method, ...)
Usage: python scripts/segment_units.py [--include-coughvid-padding] [--force]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from auscultrag.segmentation.base import ANNOTATION_SOURCES, UNIT_BOUNDS_SEC, get_segmenter, segment_row  # noqa: E402
from auscultrag.stage import cached, write_sidecar  # noqa: E402

MAX_UNITS = 16
KEEP_COLS = ["uid", "corpus", "family", "patient_id", "recording_id", "dup_group_id", "split", "fold", "duration_sec"]


def method_name(row) -> str:
    if row["corpus"] in ANNOTATION_SOURCES:
        return ANNOTATION_SOURCES[row["corpus"]][1]
    seg = get_segmenter(row["family"])
    return f"{seg.method}_v{seg.version}"


def work(row: dict) -> list[dict]:
    try:
        res = segment_row(row)
        units = [(u.start_sec, u.end_sec) for u in res.units]
        status = res.status
    except Exception as exc:  # recorded, never silent
        units, status = [], f"error:{type(exc).__name__}"
    method = method_name(row)
    if not units:
        hi = UNIT_BOUNDS_SEC[row["family"]][1]
        units, method = [(0.0, min(float(row["duration_sec"]), hi))], method + "+whole_recording_fallback"
    if len(units) > MAX_UNITS:
        idx = np.linspace(0, len(units) - 1, MAX_UNITS).round().astype(int)
        units = [units[i] for i in sorted(set(idx))]
    base = {k: row[k] for k in KEEP_COLS}
    return [{**base, "unit_id": f"{row['uid']}#{i}", "unit_index": i, "start_sec": float(s), "end_sec": float(e),
             "unit_method": method, "segment_status": status} for i, (s, e) in enumerate(units)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-coughvid-padding", action="store_true",
                    help="also segment the unlabelled COUGHVID KB-padding recordings (~31k)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    m = pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest.parquet")
    keep = m.has_closed_set_label.fillna(False) | ((m.corpus == "coughvid") & (m.split == "test"))
    if a.include_coughvid_padding:
        keep |= m.corpus == "coughvid"
    m = m[keep]
    out = ARTIFACT_ROOT / "units" / "units.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"max_units": MAX_UNITS, "bounds": UNIT_BOUNDS_SEC, "n_recordings": len(m),
              "coughvid_padding": a.include_coughvid_padding, "version": 1}
    if cached(out, config) and not a.force:
        print(f"cached: {out}")
        return
    rows = m.to_dict("records")
    units = []
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        for i, us in enumerate(pool.map(work, rows, chunksize=16), 1):
            units.extend(us)
            if i % 5000 == 0:
                print(f"segmented {i}/{len(rows)} recordings, {len(units)} units", flush=True)
    df = pd.DataFrame(units)
    df.to_parquet(out, index=False)
    summary = df.groupby(["corpus", "family", "unit_method"]).agg(
        recordings=("uid", "nunique"), units=("unit_id", "size"),
        dur_median=("end_sec", lambda s: float(np.median(s - df.loc[s.index, "start_sec"])))).round(3)
    write_sidecar(out, stage="segment_units", tier="cpu", config=config, extra={"rows": len(df)})
    print(summary.to_string())
    print(f"wrote {out}: {len(df)} units from {df.uid.nunique()} recordings")


if __name__ == "__main__":
    main()
