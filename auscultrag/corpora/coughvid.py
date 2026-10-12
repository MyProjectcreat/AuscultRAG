"""COUGHVID v3 (Zenodo 7024894, CC-BY-4.0). Family: cough. NO closed-set ground truth (R7).

Every row has has_closed_set_label = False and label_space_id = "coughvid_none". Self-reported
status, the model-derived status_SSL, and the per-expert physician annotations (diagnosis_1..4) are
kept as meta_ columns so an expert-labelled out-of-distribution subset can be built later; they are
never written into `label`.
Identity: COUGHVID has no submitter id, only a per-recording uuid. Each uuid is treated as its own
patient; repeated submissions by one person cannot be detected from metadata (a known limitation).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from auscultrag.corpora.common import BuildReport
from auscultrag.corpora.coughvid_transcode import OUT_DIR, SRC_DIR

CORPUS = "coughvid"
EXPERTS = ("1", "2", "3", "4")


def build(raw_root: Path) -> tuple[pd.DataFrame, BuildReport]:
    rep = BuildReport(CORPUS)
    root = raw_root / "coughvid"
    meta = pd.read_csv(root / SRC_DIR / "metadata_compiled.csv", low_memory=False)
    wav_dir = root / OUT_DIR
    rows = []
    for _, r in meta.iterrows():
        uuid = str(r["uuid"])
        wav = wav_dir / f"{uuid}.wav"
        if not wav.is_file():
            rep.drop("no decodable audio (missing or transcode failed)")
            continue
        diags = [str(r[f"diagnosis_{k}"]) for k in EXPERTS if pd.notna(r.get(f"diagnosis_{k}"))]
        quals = [str(r[f"quality_{k}"]) for k in EXPERTS if pd.notna(r.get(f"quality_{k}"))]
        n_exp = len(diags)
        majority = pd.NA
        if n_exp:
            (top, cnt), *rest = Counter(diags).most_common()
            if not rest or rest[0][1] < cnt:
                majority = top
        src = next((p for p in (root / SRC_DIR / f"{uuid}{ext}" for ext in (".webm", ".ogg", ".wav")) if p.exists()), None)
        rows.append({
            "uid": f"{CORPUS}:{uuid}",
            "corpus": CORPUS,
            "family": "cough",
            "patient_id": f"{CORPUS}:{uuid}",
            "recording_id": uuid,
            "device_id": "unknown",
            "dup_group_id": f"{CORPUS}:{uuid}",
            "filepath": str(wav),
            "label": pd.NA,
            "label_space_id": "coughvid_none",
            "has_closed_set_label": False,
            "split": pd.NA,
            "meta_source_path": str(src) if src else pd.NA,
            "meta_cough_detected": r["cough_detected"],
            "meta_status_self_reported": r["status"],
            "meta_status_ssl_model_derived": r["status_SSL"],
            "meta_expert_n": n_exp,
            "meta_expert_diagnoses": "|".join(diags) if diags else pd.NA,
            "meta_expert_majority": majority,
            "meta_expert_quality": "|".join(quals) if quals else pd.NA,  # good / ok / poor / no_cough
            "meta_age": r["age"],
            "meta_sex": r["gender"],
        })
    return pd.DataFrame(rows), rep
