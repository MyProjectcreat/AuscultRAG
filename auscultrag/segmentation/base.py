"""Auscultation-unit segmentation interface.

A segmenter maps one recording to a list of units (start/end seconds within the recording). The
unit contract per family (R9) is defined here so all corpora are cut to the same physiological unit:

  cardiac      one S1->S1 cycle (S1, systole, S2, diastole). Plausible 0.25-2.0 s (30-240 bpm).
  respiratory  one breathing cycle (inspiration + expiration). Plausible 0.5-10.0 s (lower bound set by
               SPRSound infants, who breathe up to ~60/min).
  cough        one cough burst (explosive onset to end of audible decay). Plausible 0.10-1.5 s.

Units outside the plausible range are dropped by the segmenter (counted, never silently). Each
segmenter declares method + version so unit tables are cache-keyed and reproducible (R13).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import soundfile as sf

UNIT_BOUNDS_SEC = {"cardiac": (0.25, 2.0), "respiratory": (0.5, 10.0), "cough": (0.10, 1.5)}


@dataclass(frozen=True)
class Unit:
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class SegmentationResult:
    units: list[Unit]
    n_candidates: int          # units proposed before the plausibility filter
    status: str                # "ok" | "no_units" | "too_short" | ...
    diagnostics: dict


class Segmenter(Protocol):
    family: str
    method: str
    version: int

    def segment(self, audio: np.ndarray, sr: int) -> SegmentationResult: ...


def load_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def filter_plausible(bounds: list[tuple[float, float]], family: str) -> tuple[list[Unit], int]:
    lo, hi = UNIT_BOUNDS_SEC[family]
    kept = [(s, e) for s, e in bounds if lo <= e - s <= hi]
    return [Unit(i, float(s), float(e)) for i, (s, e) in enumerate(kept)], len(bounds)


def get_segmenter(family: str) -> Segmenter:
    """Audio-only segmenter for a family (works on any corpus)."""
    if family == "cardiac":
        from auscultrag.segmentation.cardiac import CardiacSegmenter
        return CardiacSegmenter()
    if family == "cough":
        from auscultrag.segmentation.cough import CoughSegmenter
        return CoughSegmenter()
    if family == "respiratory":
        return FixedWindowSegmenter()
    raise NotImplementedError(f"no audio-only segmenter for family '{family}'")


class FixedWindowSegmenter:
    """Stop-gap respiratory units for corpora without cycle annotations (KAUH, Coswara breathing):
    consecutive 3-s windows (a typical adult breath), last partial window kept only if >= 1.5 s.
    These are NOT physiological cycles; rows carry method "fixed_window_3s" so analyses can separate
    them. To be replaced by a cycle detector validated against ICBHI / SPRSound annotations."""
    family = "respiratory"
    method = "fixed_window_3s"
    version = 1
    WIN = 3.0

    def segment(self, audio: np.ndarray, sr: int) -> SegmentationResult:
        dur = len(audio) / sr
        bounds, t = [], 0.0
        while t < dur:
            e = min(t + self.WIN, dur)
            if e - t >= self.WIN / 2:
                bounds.append((t, e))
            t += self.WIN
        units, n = filter_plausible(bounds, "respiratory")
        return SegmentationResult(units, n, "ok" if units else "too_short", {"method": self.method})


# Corpora whose own expert segmentation defines the units (preferred over the audio-only method).
ANNOTATION_SOURCES = {
    "circor": ("meta_segmentation_tsv", "circor_cycles"),
    "icbhi": ("meta_cycle_annotation", "icbhi_cycles"),
    "sprsound": ("meta_event_annotation", "sprsound_events"),
}


def segment_row(row) -> SegmentationResult:
    """Segment one manifest row: corpus annotations where they exist, else the family's audio method."""
    corpus, family = row["corpus"], row["family"]
    if corpus in ANNOTATION_SOURCES:
        from auscultrag.segmentation import annotations
        column, func = ANNOTATION_SOURCES[corpus]
        return getattr(annotations, func)(row[column])
    audio, sr = load_mono(row["filepath"])
    return get_segmenter(family).segment(audio, sr)


def segment_recording(filepath: str, family: str) -> SegmentationResult:
    audio, sr = load_mono(filepath)
    return get_segmenter(family).segment(audio, sr)
