"""SPRSound (SJTU paediatric respiratory sounds, CC-BY-4.0). Family: respiratory.

All BioCAS 2022-2025 releases are pooled and re-split by patient (the challenge's own train/test
splits are ignored; patient numbers do not overlap across releases, verified 2026-10-09).
Filename: <patient>_<age>_<sex 0/1>_<location p1-p8>_<recording no.>. Chest stethoscope (Yunting II).
Label (sprsound_adv4, recording level): `record_annotation` where present. 2024/2025 test files carry
only event annotations; their record label is derived with the dataset's own definition (Rhonchi,
Wheeze, Stridor = CAS; Coarse/Fine Crackle = DAS; Wheeze+Crackle = both). On the 3,324 files that
carry both annotations the derivation matches the record label 100 %. Files with no events are
exactly the "Poor Quality" records (230/230) and are excluded.
Patient diagnosis from the Patient Summary CSVs is kept as meta_diagnosis.
Dup groups: one per recording (distinct recordings at different chest locations).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from auscultrag.corpora.common import BuildReport

CORPUS = "sprsound"
CAS = {"Rhonchi", "Wheeze", "Stridor"}
DAS = {"Coarse Crackle", "Fine Crackle"}
RECORD_MAP = {"Normal": "Normal", "CAS": "CAS", "DAS": "DAS", "CAS & DAS": "CAS&DAS"}


def derive(events: list[dict]) -> str | None:
    types = {e["type"] for e in events}
    if not types:
        return None
    c = bool(types & CAS) or "Wheeze+Crackle" in types
    d = bool(types & DAS) or "Wheeze+Crackle" in types
    return "CAS&DAS" if c and d else "CAS" if c else "DAS" if d else "Normal"


def build(raw_root: Path) -> tuple[pd.DataFrame, BuildReport]:
    rep = BuildReport(CORPUS)
    root = raw_root / "sprsound" / "SPRSound"
    jsons = {p.stem: p for p in root.glob("BioCAS20*/**/*.json")}
    summ = []
    for f in (root / "Patient Summary").glob("*.csv"):
        s = pd.read_csv(f, dtype=str)
        summ.append(s[["patient_num", "disease"]])
    diag = pd.concat(summ).dropna().drop_duplicates()
    conflicts = diag.groupby("patient_num").disease.nunique()
    rep.note(f"patients with conflicting diagnoses across summary files: {int((conflicts > 1).sum())}")
    diag_map = diag.groupby("patient_num").disease.agg(lambda s: "|".join(sorted(set(s.str.strip())))).to_dict()

    rows, derived = [], 0
    for wav in sorted(root.glob("BioCAS20*/**/*.wav")):
        parts = wav.stem.split("_")
        if len(parts) != 5:
            rep.drop("filename does not have 5 fields")
            continue
        pid, age, sex, loc, recno = parts
        j = jsons.get(wav.stem)
        if j is None:
            rep.drop("no annotation json")
            continue
        ann = json.loads(j.read_text())
        rec = ann.get("record_annotation")
        if rec == "Poor Quality":
            rep.drop("record annotated Poor Quality")
            continue
        if rec is not None:
            label, source = RECORD_MAP.get(rec), "record_annotation"
        else:
            label, source = derive(ann.get("event_annotation") or []), "derived_from_events"
            derived += label is not None
        if label is None:
            rep.drop("no events and no record label (Poor Quality by construction)")
            continue
        rows.append({
            "uid": f"{CORPUS}:{wav.stem}",
            "corpus": CORPUS,
            "family": "respiratory",
            "patient_id": f"{CORPUS}:{pid}",
            "recording_id": wav.stem,
            "device_id": "yunting_ii",
            "dup_group_id": f"{CORPUS}:{wav.stem}",
            "filepath": str(wav),
            "label": label,
            "label_space_id": "sprsound_adv4",
            "has_closed_set_label": True,
            "split": pd.NA,
            "meta_label_source": source,
            "meta_release": wav.parts[len(root.parts)],
            "meta_site": loc,
            "meta_age": age,
            "meta_sex": {"0": "M", "1": "F"}.get(sex, sex),
            "meta_diagnosis": diag_map.get(pid),
            "meta_event_annotation": str(j),
        })
    rep.note(f"{derived} record labels derived from event annotations (validated 100% agreement)")
    return pd.DataFrame(rows), rep
