"""Coswara (iiscleap/Coswara-Data, CC BY 4.0). Families: cough (cough-heavy, cough-shallow) and
respiratory (breathing-deep, breathing-shallow). Vowels and counting are excluded.

Identity: one directory per participant id under Extracted_data/<date>/<id>/ (verified unique across
date folders). Participants flagged rU == 'y' (returning user) submitted again under a NEW id that
cannot be linked to their earlier id; to avoid an unlinkable same-person pair straddling splits they
are excluded. Label: covid_status (8 classes) from combined_data.csv.
Dup groups: cough-heavy vs cough-shallow (and deep vs shallow breathing) are distinct events from one
participant, so dup_group_id = recording_id; same-participant handling is via patient_id.
Quality: the dataset's manual annotation (2 excellent / 1 good / 0 bad) is kept; quality 0 and
empty (header-only) files are excluded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from auscultrag.config import LABEL_SPACES
from auscultrag.corpora.common import BuildReport

CORPUS = "coswara"
MODALITIES = {
    "cough-heavy": "cough",
    "cough-shallow": "cough",
    "breathing-deep": "respiratory",
    "breathing-shallow": "respiratory",
}
EMPTY_WAV_BYTES = 44  # RIFF header only


def build(raw_root: Path) -> tuple[pd.DataFrame, BuildReport]:
    rep = BuildReport(CORPUS)
    root = raw_root / "coswara"
    meta = pd.read_csv(root / "Coswara-Data" / "combined_data.csv")
    meta = meta.set_index("id")
    quality: dict[tuple[str, str], int] = {}
    for mod in MODALITIES:
        q = pd.read_csv(root / "Coswara-Data" / "annotations" / f"{mod}_labels.csv", skipinitialspace=True)
        for fname, val in zip(q["FILENAME"], q["QUALITY"]):
            pid, m = str(fname).rsplit("_", 1)
            quality[(pid, m)] = int(val)

    space = set(LABEL_SPACES["coswara_covid8"])
    dirs = {p.name: p for d in sorted((root / "Extracted_data").iterdir()) if d.is_dir() for p in d.iterdir() if p.is_dir()}
    returning = set(meta.index[meta["rU"] == "y"])
    rep.note(f"{len(returning)} participants flagged returning user (rU=='y') excluded; "
             f"{int(meta['rU'].isna().sum())} have rU missing and are kept")

    rows = []
    for pid, pdir in sorted(dirs.items()):
        if pid in returning:
            rep.drop("returning user (unlinkable duplicate identity)", sum((pdir / f"{m}.wav").exists() for m in MODALITIES))
            continue
        status = meta.at[pid, "covid_status"]
        if status not in space:
            rep.drop(f"label outside space: {status}")
            continue
        mj = pdir / "metadata.json"
        device = "unknown"
        if mj.exists():
            device = str(json.loads(mj.read_text()).get("dT") or "unknown")  # web / android / ios
        for mod, family in MODALITIES.items():
            wav = pdir / f"{mod}.wav"
            if not wav.exists():
                rep.drop(f"{mod}: file missing")
                continue
            if wav.stat().st_size <= EMPTY_WAV_BYTES:
                rep.drop(f"{mod}: empty file (header only)")
                continue
            qv = quality.get((pid, mod))
            if qv == 0:
                rep.drop(f"{mod}: annotated quality 0 (bad)")
                continue
            rec_id = f"{pid}_{mod}"
            rows.append({
                "uid": f"{CORPUS}:{rec_id}",
                "corpus": CORPUS,
                "family": family,
                "patient_id": f"{CORPUS}:{pid}",
                "recording_id": rec_id,
                "device_id": device,
                "dup_group_id": f"{CORPUS}:{rec_id}",
                "filepath": str(wav),
                "label": status,
                "label_space_id": "coswara_covid8",
                "has_closed_set_label": True,
                "split": pd.NA,
                "meta_modality": mod,
                "meta_quality": qv,
                "meta_age": meta.at[pid, "a"],
                "meta_sex": meta.at[pid, "g"],
                "meta_country": meta.at[pid, "l_c"],
                "meta_record_date": meta.at[pid, "record_date"],
                "meta_test_status": meta.at[pid, "test_status"],
                "meta_date_folder": pdir.parent.name,
            })
    return pd.DataFrame(rows), rep
