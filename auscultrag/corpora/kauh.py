"""KAUH lung sounds (Fraiwan et al., Mendeley Data 10.17632/jwyy9np4gv.3, CC BY 4.0). Family: respiratory.

Filename: <filter B|D|E>P<patient>_<diagnosis>,<sound type>,<location>,<age>,<sex>.wav. Every patient
has exactly one recording exported with three stethoscope filters (Bell 20-200 Hz, Diaphragm
100-500 Hz, Extended 50-500 Hz). The three files are the same physical recording (envelope
cross-correlation B vs D: median 0.944, vs 0.312 between patients), so only the Extended-range
export is kept; keeping all three would triple-count one event.
Label (kauh_adv3) from the sound-type field: W -> CAS, C/Crep -> DAS, N -> Normal. Recordings with both
wheezes and crackles (CAS&DAS: 2 patients, too few to evaluate as a class) and "Bronchial" alone are
excluded. Diagnosis kept as meta_diagnosis (case-normalised).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from auscultrag.corpora.common import BuildReport

CORPUS = "kauh"
NAME_RE = re.compile(r"^(?P<flt>[BDE])P(?P<pid>\d+)_(?P<diag>[^,]+),(?P<snd>[^,]+),(?P<loc>[^,]+),(?P<age>[^,]*),(?P<sex>[^,]*)$")
KEEP_FILTER = "E"


def adv4(sound: str) -> str | None:
    toks = sound.replace(",", " ").split()
    if toks == ["N"]:
        return "Normal"
    w = "W" in toks
    c = "C" in toks or any(t.lower().startswith("crep") for t in toks)
    if w and c:
        return "CAS&DAS"
    if w:
        return "CAS"
    if c:
        return "DAS"
    return None


def build(raw_root: Path) -> tuple[pd.DataFrame, BuildReport]:
    rep = BuildReport(CORPUS)
    rows = []
    for wav in sorted((raw_root / "kauh" / "audio").rglob("*.wav")):
        m = NAME_RE.match(wav.stem)
        if not m:
            rep.drop("unparseable file name")
            continue
        if m["flt"] != KEEP_FILTER:
            rep.drop(f"filter variant {m['flt']} of the same recording (kept {KEEP_FILTER} only)")
            continue
        label = adv4(m["snd"])
        if label is None or label == "CAS&DAS":
            rep.drop(f"sound type outside adv3 space: {m['snd'].strip()}")
            continue
        pid = int(m["pid"])
        rec_id = f"P{pid}"
        rows.append({
            "uid": f"{CORPUS}:{rec_id}",
            "corpus": CORPUS,
            "family": "respiratory",
            "patient_id": f"{CORPUS}:{pid}",
            "recording_id": rec_id,
            "device_id": "electronic_stethoscope",
            "dup_group_id": f"{CORPUS}:{rec_id}",
            "filepath": str(wav),
            "label": label,
            "label_space_id": "kauh_adv3",
            "has_closed_set_label": True,
            "split": pd.NA,
            "meta_sound_type": m["snd"].strip(),
            "meta_diagnosis": m["diag"].strip().lower(),
            "meta_site": m["loc"].strip(),
            "meta_age": m["age"].strip(),
            "meta_sex": m["sex"].strip(),
            "meta_filter": m["flt"],
        })
    return pd.DataFrame(rows), rep
