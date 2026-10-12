"""Structured diagnostic-report fields per task, with ground truth taken from each corpus's own expert
annotations. The LLM writes a report (free-text findings + impression + these fields); every field is
scored against the annotation. Fields only use annotations that describe the SOUND or the diagnosis.

Ground truth values of None mean "not annotated for this recording" and are excluded from that field's
score (e.g. CirCor murmur details when the murmur status is Unknown).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from auscultrag.config import RAW_ROOT

YN = ("yes", "no")

# task -> ordered {field: (allowed values, description shown to the model)}
FIELDS: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {
    "circor_outcome": {
        "outcome": (("Normal", "Abnormal"), "overall clinical outcome"),
        "murmur": (("Present", "Absent", "Unknown"), "is a heart murmur present in this patient"),
        "murmur_audible_here": (YN, "is the murmur audible in THIS recording's auscultation site"),
        "systolic_timing": (("none", "Early-systolic", "Mid-systolic", "Late-systolic", "Holosystolic"), "timing of the systolic murmur"),
        "systolic_shape": (("none", "Crescendo", "Decrescendo", "Diamond", "Plateau"), "shape of the systolic murmur"),
        "systolic_grading": (("none", "I/VI", "II/VI", "III/VI"), "grade of the systolic murmur"),
        "systolic_pitch": (("none", "Low", "Medium", "High"), "pitch of the systolic murmur"),
        "systolic_quality": (("none", "Blowing", "Harsh", "Musical"), "quality of the systolic murmur"),
    },
    "icbhi_chronic": {
        "condition": (("Chronic", "NonChronic", "Healthy"), "Chronic (COPD, bronchiectasis, asthma), NonChronic (infections), or Healthy"),
        "diagnosis": (("COPD", "Bronchiectasis", "Asthma", "URTI", "LRTI", "Pneumonia", "Bronchiolitis", "Healthy"), "specific diagnosis"),
        "crackles": (YN, "are crackles present in this recording"),
        "wheezes": (YN, "are wheezes present in this recording"),
    },
    "icbhi_adventitious": {
        "adventitious": (("Normal", "CAS", "DAS", "CAS&DAS"), "adventitious sounds: none, continuous (wheeze), discontinuous (crackle), both"),
        "crackles": (YN, "are crackles present in this recording"),
        "wheezes": (YN, "are wheezes present in this recording"),
        "condition": (("Chronic", "NonChronic", "Healthy"), "Chronic (COPD, bronchiectasis, asthma), NonChronic (infections), or Healthy"),
    },
    "sprsound_adventitious": {
        "adventitious": (("Normal", "CAS", "DAS", "CAS&DAS"), "adventitious sounds: none, continuous, discontinuous, both"),
        "crackles": (YN, "fine or coarse crackles present"),
        "wheeze": (YN, "wheeze present"),
        "rhonchi": (YN, "rhonchi present"),
        "stridor": (YN, "stridor present"),
        "diagnosis": (("Pneumonia", "Bronchitis", "Asthma", "Control", "Other"), "patient diagnosis group"),
    },
    "kauh_adventitious": {
        "adventitious": (("Normal", "CAS", "DAS"), "adventitious sounds: none, continuous (wheeze), discontinuous (crackle)"),
        "wheezes": (YN, "wheezes present"),
        "crackles": (YN, "crackles present"),
        "diagnosis": (("Normal", "Asthma", "COPD", "Heart failure", "Pneumonia", "Lung fibrosis", "Other"), "patient diagnosis group"),
    },
    "tbscreen_cough": {"tb_status": (("TB", "NTB"), "pulmonary tuberculosis (TB) or another respiratory illness (NTB)")},
    "coswara_cough": {"covid_status": (("healthy", "no_resp_illness_exposed", "positive_asymp", "positive_mild", "positive_moderate",
                                         "recovered_full", "resp_illness_not_identified", "under_validation"), "COVID-19 status")},
    "coswara_respiratory": {"covid_status": (("healthy", "no_resp_illness_exposed", "positive_asymp", "positive_mild", "positive_moderate",
                                               "recovered_full", "resp_illness_not_identified", "under_validation"), "COVID-19 status")},
}
PRIMARY_FIELD = {t: next(iter(f)) for t, f in FIELDS.items()}


@lru_cache(maxsize=1)
def _circor_meta() -> pd.DataFrame:
    p = RAW_ROOT / "circor" / "circor-heart-sound-1.0.3" / "training_data.csv"
    return pd.read_csv(p, dtype=str).set_index("Patient ID")


def _norm_none(v) -> str:
    return "none" if pd.isna(v) or str(v).strip() in ("", "nan") else str(v).strip()


def circor_gt(row: pd.Series) -> dict:
    m = _circor_meta().loc[str(row["meta_native_patient_id"])]
    out = {"outcome": row["label"], "murmur": m["Murmur"]}
    if m["Murmur"] == "Unknown":
        return out | {k: None for k in FIELDS["circor_outcome"] if k not in out}
    locs = set(str(m["Murmur locations"]).split("+")) if m["Murmur"] == "Present" else set()
    site = str(row["meta_site"]).split("_")[0]
    out["murmur_audible_here"] = "yes" if site in locs else "no"
    for f, col in (("systolic_timing", "Systolic murmur timing"), ("systolic_shape", "Systolic murmur shape"),
                   ("systolic_grading", "Systolic murmur grading"), ("systolic_pitch", "Systolic murmur pitch"),
                   ("systolic_quality", "Systolic murmur quality")):
        v = _norm_none(m[col])
        out[f] = v if v in FIELDS["circor_outcome"][f][0] else None
    return out


def _icbhi_cycles(path: str) -> tuple[str, str]:
    a = pd.read_csv(path, sep=r"\s+", header=None, names=["s", "e", "c", "w"])
    return ("yes" if a.c.astype(int).any() else "no"), ("yes" if a.w.astype(int).any() else "no")


def icbhi_gt(row: pd.Series, task: str) -> dict:
    c, w = _icbhi_cycles(row["meta_cycle_annotation"])
    base = {"crackles": c, "wheezes": w, "condition": row["label"]}
    if task == "icbhi_chronic":
        return base | {"diagnosis": row["meta_diagnosis_native"]}
    return base | {"adventitious": row["meta_label_adv4"]}


SPR_CAS = {"Wheeze": "wheeze", "Rhonchi": "rhonchi", "Stridor": "stridor"}


def _spr_diag(d) -> str | None:
    if pd.isna(d):
        return None
    d = str(d).lower()
    if "|" in d:
        return None  # conflicting diagnoses across summary files
    if "pneumonia" in d:
        return "Pneumonia"
    if "bronchitis" in d:
        return "Bronchitis"
    if "asthma" in d:
        return "Asthma"
    if "control" in d:
        return "Control"
    return "Other"


def sprsound_gt(row: pd.Series) -> dict:
    ev = json.loads(Path(row["meta_event_annotation"]).read_text()).get("event_annotation") or []
    types = {e["type"] for e in ev}
    both = "Wheeze+Crackle" in types
    return {"adventitious": row["label"],
            "crackles": "yes" if (types & {"Fine Crackle", "Coarse Crackle"}) or both else "no",
            "wheeze": "yes" if "Wheeze" in types or both else "no",
            "rhonchi": "yes" if "Rhonchi" in types else "no",
            "stridor": "yes" if "Stridor" in types else "no",
            "diagnosis": _spr_diag(row.get("meta_diagnosis"))}


def _kauh_diag(d: str) -> str:
    d = str(d).lower()
    if d in ("n", "normal"):
        return "Normal"
    if "+" in d or " and " in d:
        return "Other"
    for key, name in (("asthma", "Asthma"), ("copd", "COPD"), ("heart failure", "Heart failure"),
                      ("pneumonia", "Pneumonia"), ("lung fibrosis", "Lung fibrosis")):
        if key in d:
            return name
    return "Other"


def kauh_gt(row: pd.Series) -> dict:
    toks = str(row["meta_sound_type"]).replace(",", " ").split()
    return {"adventitious": row["label"],
            "wheezes": "yes" if "W" in toks else "no",
            "crackles": "yes" if ("C" in toks or any(t.lower().startswith("crep") for t in toks)) else "no",
            "diagnosis": _kauh_diag(row["meta_diagnosis"])}


def ground_truth(task: str, row: pd.Series) -> dict:
    if task == "circor_outcome":
        return circor_gt(row)
    if task.startswith("icbhi"):
        return icbhi_gt(row, task)
    if task == "sprsound_adventitious":
        return sprsound_gt(row)
    if task == "kauh_adventitious":
        return kauh_gt(row)
    return {PRIMARY_FIELD[task]: row["label"]}
