"""Units derived from a corpus's own expert segmentation (CirCor heart cycles, ICBHI respiratory
cycles, SPRSound respiratory events).

CirCor: every .tsv row is (start, end, state) with 1 = S1, 2 = systole, 3 = S2, 4 = diastole and
0 = unannotated (one file also contains the out-of-spec state 28). A unit is S1 onset -> next S1
onset, emitted only when the four intervening states are exactly [1, 2, 3, 4] and contiguous, so
partially annotated or malformed stretches never produce a unit.
"""
from __future__ import annotations

import json

import pandas as pd

from auscultrag.segmentation.base import SegmentationResult, filter_plausible

CIRCOR_METHOD = ("circor_expert_annotation", 1)


def circor_cycles(tsv_path: str) -> SegmentationResult:
    a = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "state"])
    starts, states, ends = a["start"].to_numpy(), a["state"].to_numpy(), a["end"].to_numpy()
    s1 = [i for i in range(len(a)) if states[i] == 1]
    bounds = []
    for i, j in zip(s1[:-1], s1[1:]):
        contiguous = all(abs(ends[k] - starts[k + 1]) < 1e-6 for k in range(i, j))
        if j - i == 4 and list(states[i:j]) == [1, 2, 3, 4] and contiguous:
            bounds.append((float(starts[i]), float(starts[j])))
    units, n_cand = filter_plausible(bounds, "cardiac")
    return SegmentationResult(units, n_cand, "ok" if units else "no_units",
                              {"method": CIRCOR_METHOD[0], "out_of_spec_states": sorted(set(states) - {0, 1, 2, 3, 4})})


def icbhi_cycles(txt_path: str) -> SegmentationResult:
    """ICBHI: each .txt row is one expert-marked respiratory cycle (start, end, crackles, wheezes)."""
    a = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["start", "end", "crackles", "wheezes"])
    bounds = [(float(s), float(e)) for s, e in zip(a.start, a.end)]
    units, n_cand = filter_plausible(bounds, "respiratory")
    return SegmentationResult(units, n_cand, "ok" if units else "no_units", {"method": "icbhi_expert_cycles"})


def sprsound_events(json_path: str) -> SegmentationResult:
    """SPRSound: event_annotation rows (ms) mark individual respiratory events (one breath in these
    paediatric recordings; median 1.7 s vs 2.5 s ICBHI adult cycles). Overlapping events occur in the
    source annotation and are kept as given (counted in diagnostics)."""
    ev = json.loads(open(json_path).read()).get("event_annotation") or []
    bounds = sorted((int(e["start"]) / 1000.0, int(e["end"]) / 1000.0) for e in ev)
    overlaps = sum(1 for (s0, e0), (s1, _) in zip(bounds, bounds[1:]) if s1 < e0)
    units, n_cand = filter_plausible(bounds, "respiratory")
    return SegmentationResult(units, n_cand, "ok" if units else "no_units",
                              {"method": "sprsound_expert_events", "overlapping_pairs": overlaps})
