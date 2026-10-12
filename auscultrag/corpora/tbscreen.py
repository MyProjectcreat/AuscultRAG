"""TBScreen (Sharma et al., Science Advances 2024; Zenodo 10431329, CC-BY-4.0). Family: cough.

Identity (verified from Metadata.csv / Dataset_key.xlsx, not filenames):
  * The person key is (cohort, sub) from the per-cough CSV, NOT the `subject` string: Metadata.csv
    has two different people under subject "PID_4A" (sub 4, TB, and sub 110, NTB).
  * Cohort A and cohort C reuse sub numbers for different people (ages/sex differ), so A10 != C10.
  * Exception: "PID_32A_now_C" (A32) and "PID_32C" (C32) agree on age, sex, GeneXpert, culture, smear,
    HIV, smoking and cough duration; they cannot be proven distinct, so they are conservatively merged
    into one patient (over-merging can only make the split stricter).
Near-duplicates (R3): each physical cough was captured by up to three microphones (pixel, codec,
yeti); coughs longer than 1 s were cut into consecutive 1-s files suffixed _0, _1, ... (see
audio_pre-processing/cough_processing.py::length_trim in the authors' code). The cough index in the
filename is shared across devices (a cross-device audio probe matched same-index files rank-1 in
63/70 cases). dup_group_id = (patient, passive|forced, cough index): all devices and all 1-s pieces of
one cough share a group.
Consent: rows with Permission_sound != "Yes" are excluded even when a wav was shipped.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from auscultrag.corpora.common import BuildReport

CORPUS = "tbscreen"
ROOT_REL = "tbscreen/TBscreen_Dataset"
KINDS = {"passive": ("Passive_coughs", "Passive_coughs.csv"), "forced": ("Forced_coughs", "Forced_coughs.csv")}
PATH_RE = re.compile(r"^(?P<prefix>PID_(?P<num>\d+)[AC].*?)_?(?P<idx>\d+)_(?P<dev>pixel|codec|yeti)(?:_(?P<seg>\d+))?$")
PERSON_ALIASES = {"A32": "C32"}  # see module docstring


def person_key(cohort: str, sub: int) -> str:
    key = f"{cohort}{int(sub)}"
    return PERSON_ALIASES.get(key, key)


def build(raw_root: Path) -> tuple[pd.DataFrame, BuildReport]:
    rep = BuildReport(CORPUS)
    root = raw_root / ROOT_REL
    rep.note("person aliases merged: " + ", ".join(f"{a}->{b}" for a, b in PERSON_ALIASES.items()))
    rows = []
    for kind, (folder, csv) in KINDS.items():
        df = pd.read_csv(root / folder / csv, low_memory=False)
        audio_dir = root / folder / "Audio_files"
        for _, r in df.iterrows():
            stem = str(r["path"])
            wav = audio_dir / f"{stem}.wav"
            has_wav = wav.is_file()
            if str(r["Permission_sound"]) != "Yes":
                rep.drop(f"{kind}: Permission_sound != Yes" + (" (wav shipped anyway)" if has_wav else ""))
                continue
            if not has_wav:
                rep.drop(f"{kind}: consented row but wav absent from archive")
                continue
            if pd.isna(r["cohort"]) or pd.isna(r["sub"]):
                rep.drop(f"{kind}: cohort/sub missing")
                continue
            m = PATH_RE.match(stem)
            if not m:
                rep.drop(f"{kind}: unparseable file name")
                continue
            if int(m["num"]) != int(r["sub"]) or m["dev"] != r["device"]:
                rep.drop(f"{kind}: file name disagrees with sub/device columns")
                continue
            person = person_key(str(r["cohort"]), int(r["sub"]))
            cough_idx = int(m["idx"])
            seg = int(m["seg"]) if m["seg"] is not None else -1  # -1: short cough, centred and padded
            rows.append({
                "uid": f"{CORPUS}:{kind}:{stem}",
                "corpus": CORPUS,
                "family": "cough",
                "patient_id": f"{CORPUS}:{person}",
                "recording_id": f"{kind}:{stem}",
                "device_id": m["dev"],
                "dup_group_id": f"{CORPUS}:{person}:{kind}:{cough_idx}",
                "filepath": str(wav),
                "label": str(r["Label"]),
                "label_space_id": "tbscreen_tb2",
                "has_closed_set_label": True,
                "split": pd.NA,
                "meta_cough_type": kind,
                "meta_cough_index": cough_idx,
                "meta_segment_index": seg,
                "meta_native_subject": r["subject"],
                "meta_cohort": r["cohort"],
                "meta_age": r["age"],
                "meta_sex": r["gndr_a"],
                "meta_genexpert": r.get("lab_sptm_xprt_rslt_a"),
                "meta_hiv": r.get("hiv_infected_a"),
            })
    out = pd.DataFrame(rows)
    if len(out):
        labels_per_person = out.groupby("patient_id")["label"].nunique()
        rep.note(f"patients with >1 label: {int((labels_per_person > 1).sum())}")
    return out, rep
