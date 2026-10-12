"""Cough burst segmentation: one unit = one cough burst (explosive onset to end of audible decay).

Method (energy with hysteresis):
  1. Frame RMS, 25 ms window / 10 ms hop, in dB.
  2. Noise floor = 10th percentile of frame dB; peak = maximum.
  3. Onset threshold = floor + max(MIN_RISE_DB, 0.4 * (peak - floor)); offset threshold halfway
     between floor and onset threshold (hysteresis avoids chattering on the decay).
  4. Active runs separated by < MERGE_GAP_SEC are merged (a single cough's phases can dip briefly).
  5. Runs outside the plausible cough duration (0.10-1.5 s) are dropped and counted.
Failure modes: back-to-back cough trains with no silent gap (merged into one over-long run, then
dropped); loud background events or speech (false units); very quiet coughs below the adaptive
threshold when a louder event in the same recording sets the peak; clips that are themselves cut
from a longer cough (TBScreen pieces _1, _2, ...) yield truncated units by construction.
"""
from __future__ import annotations

import numpy as np

from auscultrag.segmentation.base import SegmentationResult, filter_plausible

WIN_SEC, HOP_SEC = 0.025, 0.010
MIN_RISE_DB = 10.0
REL_RISE = 0.4
MERGE_GAP_SEC = 0.08


class CoughSegmenter:
    family = "cough"
    method = "rms_hysteresis"
    version = 1

    def segment(self, audio: np.ndarray, sr: int) -> SegmentationResult:
        win, hop = int(WIN_SEC * sr), int(HOP_SEC * sr)
        if len(audio) < win:
            return SegmentationResult([], 0, "too_short", {})
        n = 1 + (len(audio) - win) // hop
        idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
        rms = np.sqrt(np.mean(audio[idx].astype(np.float64) ** 2, axis=1))
        db = 20 * np.log10(rms + 1e-10)
        floor, peak = np.percentile(db, 10), db.max()
        on = floor + max(MIN_RISE_DB, REL_RISE * (peak - floor))
        off = floor + 0.5 * (on - floor)
        if peak < on:
            return SegmentationResult([], 0, "no_units", {"floor_db": floor, "peak_db": peak})

        runs, active, start = [], False, 0
        for i, v in enumerate(db):
            if not active and v >= on:
                active, start = True, i
            elif active and v < off:
                runs.append([start, i])
                active = False
        if active:
            runs.append([start, len(db)])
        merged = []
        for s, e in runs:
            if merged and (s - merged[-1][1]) * HOP_SEC < MERGE_GAP_SEC:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        bounds = [(s * HOP_SEC, min(len(audio) / sr, e * HOP_SEC + WIN_SEC)) for s, e in merged]
        units, n_cand = filter_plausible(bounds, "cough")
        return SegmentationResult(units, n_cand, "ok" if units else "no_units",
                                  {"floor_db": float(floor), "peak_db": float(peak), "n_runs": len(runs)})
