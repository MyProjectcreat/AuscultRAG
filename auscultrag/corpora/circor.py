"""CirCor DigiScope Phonocardiogram Dataset v1.0.3 (PhysioNet, ODC-By 1.0). Family: cardiac.

Identity: per-patient .txt files enumerate recordings; training_data.csv gives patient labels.
140 patients carry an "Additional ID" that is itself a Patient ID in this release (the same child
seen in both the 2014 and 2015 campaigns). Those IDs are merged with union-find so they form one
patient; otherwise the patient-disjoint split would be fictitious.
Dup groups: each auscultation position (AV/PV/TV/MV/Phc, plus repeated _1/_2 takes) is a distinct
recording event, so dup_group_id = recording_id; same-patient handling is via patient_id.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from auscultrag.corpora.common import BuildReport, UnionFind

CORPUS = "circor"
ROOT_REL = "circor/circor-heart-sound-1.0.3"


def build(raw_root: Path) -> tuple[pd.DataFrame, BuildReport]:
    rep = BuildReport(CORPUS)
    root = raw_root / ROOT_REL
    data_dir = root / "training_data"
    meta = pd.read_csv(root / "training_data.csv")

    uf = UnionFind()
    known = {str(p) for p in meta["Patient ID"]}
    for _, r in meta.iterrows():
        pid = str(r["Patient ID"])
        uf.find(pid)
        if pd.notna(r["Additional ID"]):
            add = str(int(r["Additional ID"]))
            if add not in known:
                rep.note(f"Additional ID {add} of patient {pid} not present in release; kept as alias only")
            uf.union(pid, add)
    groups: dict[str, list[str]] = {}
    for pid in known:
        groups.setdefault(uf.find(pid), []).append(pid)
    merged = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    rep.note(f"{len(known)} Patient IDs resolve to {len(groups)} persons; {len(merged)} persons carry 2+ IDs")
    conflicting = 0
    by_id = meta.set_index(meta["Patient ID"].astype(str))
    for members in merged.values():
        if by_id.loc[members, "Outcome"].nunique() > 1:
            conflicting += 1
    rep.note(f"{conflicting} merged persons have different Outcome labels across their campaign visits "
             f"(labels kept per visit; split is per person)")

    rows = []
    for _, r in meta.iterrows():
        pid = str(r["Patient ID"])
        txt = data_dir / f"{pid}.txt"
        lines = txt.read_text().strip().splitlines()
        header = lines[0].split()
        n_rec, fs = int(header[1]), int(header[2])
        rec_lines = [ln.split() for ln in lines[1 : 1 + n_rec]]
        for loc, hea, wav, tsv in rec_lines:
            wav_path = data_dir / wav
            if not wav_path.is_file():
                rep.drop("wav listed in patient .txt but missing on disk")
                continue
            rec_id = Path(wav).stem
            rows.append({
                "uid": f"{CORPUS}:{rec_id}",
                "corpus": CORPUS,
                "family": "cardiac",
                "patient_id": f"{CORPUS}:{uf.find(pid)}",
                "recording_id": rec_id,
                "device_id": "unknown",
                "dup_group_id": f"{CORPUS}:{rec_id}",
                "filepath": str(wav_path),
                "label": str(r["Outcome"]),
                "label_space_id": "circor_outcome2",
                "has_closed_set_label": True,
                "split": pd.NA,
                "meta_native_patient_id": pid,
                "meta_site": loc,
                "meta_murmur": r["Murmur"],
                "meta_campaign": r["Campaign"],
                "meta_age_group": r["Age"],
                "meta_sex": r["Sex"],
                "meta_pregnant": str(r["Pregnancy status"]),
                "meta_header_fs": fs,
                "meta_segmentation_tsv": str(data_dir / tsv),
            })
    return pd.DataFrame(rows), rep
