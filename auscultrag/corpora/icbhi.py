"""ICBHI 2017 Respiratory Sound Database. Family: respiratory.

Filename semantics (from the challenge description): <patient>_<recording index>_<chest location>_
<acquisition mode sc|mc>_<equipment>. Patient diagnosis from ICBHI_Challenge_diagnosis.txt.
Each file has a companion .txt with expert respiratory-cycle boundaries (start, end, crackles,
wheezes), recorded as meta_cycle_annotation for segmentation validation.
Labels: primary = icbhi_chronic3 (Chronic: COPD, Bronchiectasis, Asthma; NonChronic: URTI, LRTI,
Pneumonia, Bronchiolitis; Healthy), the grouping used in the ICBHI literature, because the native
8-way diagnosis has classes with 1-2 patients. Secondary = icbhi_adv4 recording-level adventitious
sounds aggregated from the expert cycle annotations (any wheeze -> CAS, any crackle -> DAS, both ->
CAS&DAS, none -> Normal), harmonised with SPRSound and KAUH.
Dup groups: multichannel ("mc") recordings capture several chest locations simultaneously; they are
different signals of one breathing episode from one patient, handled by patient-level disjointness
(like CirCor positions), so dup_group_id = recording_id.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from auscultrag.corpora.common import BuildReport

CORPUS = "icbhi"
CHRONIC3 = {"COPD": "Chronic", "Bronchiectasis": "Chronic", "Asthma": "Chronic",
            "URTI": "NonChronic", "LRTI": "NonChronic", "Pneumonia": "NonChronic", "Bronchiolitis": "NonChronic",
            "Healthy": "Healthy"}


def adventitious_from_cycles(ann_path: Path) -> tuple[str | None, int]:
    """Recording-level adv4 label from ICBHI cycle rows (start, end, crackles, wheezes)."""
    a = pd.read_csv(ann_path, sep=r"\s+", header=None, names=["start", "end", "crackles", "wheezes"])
    if a.empty:
        return None, 0
    c, w = bool(a.crackles.astype(int).any()), bool(a.wheezes.astype(int).any())
    return ("CAS&DAS" if c and w else "CAS" if w else "DAS" if c else "Normal"), len(a)


def build(raw_root: Path) -> tuple[pd.DataFrame, BuildReport]:
    rep = BuildReport(CORPUS)
    root = raw_root / "icbhi"
    data_dir = root / "ICBHI_final_database"
    diag = pd.read_csv(root / "ICBHI_Challenge_diagnosis.txt", sep="\t", header=None, names=["pid", "diagnosis"],
                       dtype=str)
    diag_map = dict(zip(diag.pid.str.strip(), diag.diagnosis.str.strip()))
    demo = pd.read_csv(root / "ICBHI_Challenge_demographic_information.txt", sep="\t", header=None,
                       names=["pid", "age", "sex", "adult_bmi", "child_weight", "child_height"], dtype=str)
    demo_map = demo.set_index(demo.pid.str.strip()).to_dict("index")
    official = pd.read_csv(root / "ICBHI_challenge_train_test.txt", sep="\t", header=None, names=["rec", "split"])
    official_map = dict(zip(official.rec.str.strip(), official.split.str.strip()))
    renamed = {ln.strip().strip("'") for ln in (root / "filename_differences.txt").read_text().splitlines() if ln.strip()}

    rows = []
    for wav in sorted(data_dir.glob("*.wav")):
        rec_id = wav.stem
        parts = rec_id.split("_")
        if len(parts) != 5:
            rep.drop("filename does not have 5 fields")
            continue
        pid, rec_idx, loc, mode, equip = parts
        if pid not in diag_map:
            rep.drop("patient missing from diagnosis file")
            continue
        ann = wav.with_suffix(".txt")
        d = demo_map.get(pid, {})
        adv, n_cycles = adventitious_from_cycles(ann) if ann.is_file() else (None, 0)
        if diag_map[pid] not in CHRONIC3:
            rep.drop(f"diagnosis outside chronic3 mapping: {diag_map[pid]}")
            continue
        rows.append({
            "uid": f"{CORPUS}:{rec_id}",
            "corpus": CORPUS,
            "family": "respiratory",
            "patient_id": f"{CORPUS}:{pid}",
            "recording_id": rec_id,
            "device_id": equip,
            "dup_group_id": f"{CORPUS}:{rec_id}",
            "filepath": str(wav),
            "label": CHRONIC3[diag_map[pid]],
            "label_space_id": "icbhi_chronic3",
            "has_closed_set_label": True,
            "split": pd.NA,
            "meta_diagnosis_native": diag_map[pid],
            "meta_label_adv4": adv,
            "meta_n_cycles_annotated": n_cycles,
            "meta_site": loc,
            "meta_acq_mode": mode,
            "meta_recording_index": rec_idx,
            "meta_age": d.get("age"),
            "meta_sex": d.get("sex"),
            "meta_official_split": official_map.get(rec_id),
            "meta_cycle_annotation": str(ann) if ann.is_file() else pd.NA,
        })
    n_renamed_on_disk = sum(1 for r in rows if r["recording_id"] in renamed)
    rep.note(f"filename_differences.txt lists {len(renamed)} names; {n_renamed_on_disk} of them are on-disk filenames")
    rep.note(f"{sum(1 for r in rows if r['meta_official_split'] is None)} recordings absent from the official split file")
    return pd.DataFrame(rows), rep
