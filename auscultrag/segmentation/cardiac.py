"""Cardiac cycle segmentation: one unit = S1 onset to next S1 onset.

Method (envelope + period + phase, after the classic Schmidt/Springer pipeline, without the HSMM):
  1. Band-pass 25-400 Hz (Butterworth, zero-phase), resample-free.
  2. Shannon-energy envelope, 20 ms smoothing, sampled at 100 Hz.
  3. Heart period T = earliest autocorrelation peak within 0.25-2.0 s reaching 75 % of the strongest
     peak (the global maximum is frequently a multiple of the true period).
  4. Candidate sounds = envelope peaks at least 0.15*T apart above an adaptive threshold.
  5. S1 vs S2: systole is shorter than diastole, so for each candidate pair (a, b) with b-a in
     [0.2T, 0.5T] the first is S1. S1 events are then chained greedily with spacing ~T (0.7T-1.3T).
  6. Onset of each S1 = where the envelope rises through 50 % of that peak, searched back <= 80 ms.
  7. Consecutive S1 onsets bound a unit; spacing outside 0.6T-1.4T is treated as a missed/extra beat
     and that pair is not emitted.
Failure modes: murmurs that fill systole/diastole (envelope loses the S1/S2 contrast), irregular
rhythm (fixed T assumption), low-SNR or movement artefacts, tachycardic infants where systole and
diastole durations converge (S1/S2 disambiguation degrades).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from auscultrag.segmentation.base import SegmentationResult, filter_plausible

ENV_FS = 100
PERIOD_REL_HEIGHT = 0.75


class CardiacSegmenter:
    family = "cardiac"
    method = "shannon_envelope_period_phase"
    version = 2

    def segment(self, audio: np.ndarray, sr: int) -> SegmentationResult:
        if len(audio) < 2 * sr:
            return SegmentationResult([], 0, "too_short", {})
        sos = butter(4, [25, min(400, sr / 2 - 1)], btype="bandpass", fs=sr, output="sos")
        x = sosfiltfilt(sos, audio.astype(np.float64))
        x = x / (np.max(np.abs(x)) + 1e-12)
        se = -(x ** 2) * np.log(x ** 2 + 1e-12)
        win = max(1, int(0.02 * sr))
        env = np.convolve(se, np.ones(win) / win, mode="same")
        hop = max(1, sr // ENV_FS)
        env = env[::hop]
        env = (env - env.mean()) / (env.std() + 1e-12)

        ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
        lo, hi = int(0.25 * ENV_FS), int(2.0 * ENV_FS)
        if len(ac) <= hi:
            return SegmentationResult([], 0, "too_short", {})
        # The global autocorrelation maximum often sits on a multiple of the true period; take the
        # earliest local peak that is at least PERIOD_REL_HEIGHT of the strongest one.
        seg = ac[lo:hi]
        pk, _ = find_peaks(seg)
        if len(pk) == 0:
            period = (lo + int(np.argmax(seg))) / ENV_FS
        else:
            strong = pk[seg[pk] >= PERIOD_REL_HEIGHT * seg[pk].max()]
            period = (lo + int(strong[0])) / ENV_FS

        peaks, _ = find_peaks(env, distance=max(1, int(0.15 * period * ENV_FS)), height=np.percentile(env, 60))
        t = peaks / ENV_FS
        s1 = []
        for i in range(len(t) - 1):
            gap = t[i + 1] - t[i]
            if 0.2 * period <= gap <= 0.5 * period:
                s1.append(i)
        chain = []
        for i in s1:
            # Refractory rule: an S1 closer than 0.7T to the previous one is a spurious detection.
            # Larger gaps (missed beats) are kept here and rejected as units in the spacing filter below.
            if not chain or t[i] - t[chain[-1]] >= 0.7 * period:
                chain.append(i)
        onsets = []
        for i in chain:
            p = peaks[i]
            thr = 0.5 * env[p] + 0.5 * np.median(env)
            j = p
            while j > 0 and p - j < int(0.08 * ENV_FS) and env[j - 1] > thr:
                j -= 1
            onsets.append(j / ENV_FS)
        bounds = [(a, b) for a, b in zip(onsets[:-1], onsets[1:]) if 0.6 * period <= b - a <= 1.4 * period]
        units, n_cand = filter_plausible(bounds, "cardiac")
        status = "ok" if units else "no_units"
        return SegmentationResult(units, n_cand, status, {"period_sec": period, "n_peaks": int(len(peaks)),
                                                          "n_s1": len(chain)})
