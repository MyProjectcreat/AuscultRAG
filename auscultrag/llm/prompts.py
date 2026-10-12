"""Prompt rendering. Both conditions share the SAME instructions, label enumeration and output contract;
the w/ RAG condition only appends an evidence block. (The previous implementation differed between
conditions; this is the controlled version.)"""
from __future__ import annotations

import pandas as pd

TASK_DESCRIPTIONS = {
    "coswara_cough": "a voluntary cough recording; classify the participant's COVID-19 status",
    "coswara_respiratory": "a breathing-sound recording; classify the participant's COVID-19 status",
    "tbscreen_cough": "a cough recording; classify whether the person has pulmonary tuberculosis (TB) or another respiratory illness (NTB)",
    "icbhi_chronic": "a chest auscultation recording of lung sounds; classify the patient's condition as Chronic (e.g. COPD, bronchiectasis, asthma), NonChronic (e.g. respiratory tract infection, pneumonia, bronchiolitis) or Healthy",
    "icbhi_adventitious": "a chest auscultation recording of lung sounds; classify which adventitious sounds are present: Normal (none), CAS (continuous: wheeze/rhonchi), DAS (discontinuous: crackles) or CAS&DAS (both)",
    "sprsound_adventitious": "a paediatric chest auscultation recording; classify which adventitious sounds are present: Normal (none), CAS (continuous: wheeze/rhonchi/stridor), DAS (discontinuous: crackles) or CAS&DAS (both)",
    "kauh_adventitious": "a chest auscultation recording of lung sounds; classify which adventitious sounds are present: Normal (none), CAS (continuous: wheeze) or DAS (discontinuous: crackles)",
    "circor_outcome": "a heart-sound (phonocardiogram) recording from a child; classify the clinical outcome as Normal or Abnormal",
}

# Genuine per-corpus metadata shown for retrieved cases (never the corpus name).
CASE_META = {
    "coswara": [("meta_age", "age"), ("meta_sex", "sex")],
    "tbscreen": [("meta_age", "age"), ("device_id", "microphone")],
    "icbhi": [("meta_site", "chest site"), ("device_id", "stethoscope"), ("meta_age", "age"), ("meta_sex", "sex")],
    "sprsound": [("meta_site", "chest site"), ("meta_age", "age"), ("meta_sex", "sex")],
    "kauh": [("meta_site", "chest site"), ("meta_age", "age"), ("meta_sex", "sex")],
    "circor": [("meta_site", "auscultation site"), ("meta_age_group", "age group"), ("meta_sex", "sex")],
}


def base_prompt(task: str, labels: list[str]) -> str:
    return (
        "You are assisting with research on clinical audio classification.\n"
        f"You will hear {TASK_DESCRIPTIONS[task]}.\n"
        f"Valid labels (choose exactly one): {', '.join(labels)}.\n"
        "Listen to the recording, reason briefly about the acoustic evidence, then end your reply with one line "
        "of the form:\nFinal answer: <label>\n"
        "where <label> is copied exactly from the list of valid labels."
    )


def evidence_block(cases: list[dict], labels_tbl: pd.DataFrame) -> str:
    lines = ["Reference cases retrieved from other patients by acoustic similarity "
             "(they may be wrong; use them as evidence, not as the answer):"]
    for rank, c in enumerate(cases, 1):
        row = labels_tbl.loc[c["uid"]]
        meta = [f"{name}={row[col]}" for col, name in CASE_META.get(row["corpus"], [])
                if col in row and pd.notna(row[col]) and str(row[col]) not in ("", "nan")]
        lab = c.get("label") if c.get("label") is not None else "unknown"
        lines.append(f"[{rank}] similarity={c['score']:.3f}; label={lab}" + ("; " + "; ".join(meta) if meta else ""))
    return "\n".join(lines)


def render(task: str, labels: list[str], condition: str, cases: list[dict] | None = None,
           labels_tbl: pd.DataFrame | None = None) -> str:
    p = base_prompt(task, labels)
    if condition == "rag":
        p += "\n\n" + evidence_block(cases or [], labels_tbl)
    return p


# ---------------------------------------------------------------------------------------------------
# Prompt v2 (chosen on dev, 2026-10-10). Changes vs v1, applied identically to BOTH conditions:
#   * research framing + explicit note that a (possibly very short) audio clip is attached
#     (v1 drew refusals and "please play the audio" replies, esp. on 1-s TBScreen clips);
#   * the model reports a probability for every label, so its belief can be fused with retrieval.
# w/ RAG additionally shows a similarity-weighted label distribution of the retrieved cases before the
# case list (v1 listed cases only; the model over-weighted rank 1-2 despite ~0.01 similarity gaps).
# ---------------------------------------------------------------------------------------------------

def base_prompt_v2(task: str, labels: list[str]) -> str:
    return (
        "This is a research benchmark on clinical audio classification; you are not giving medical advice.\n"
        f"An audio clip is attached (it may be short, about 1-15 seconds). It is {TASK_DESCRIPTIONS[task]}.\n"
        f"Valid labels: {', '.join(labels)}.\n"
        "Listen to the clip and reason briefly about the acoustic evidence. Then end your reply with exactly two lines:\n"
        "Probabilities: " + ", ".join(f"{l}=<p>" for l in labels) + "   (numbers between 0 and 1 that sum to 1)\n"
        "Final answer: <label>   (one label copied exactly from the valid labels)"
    )


def evidence_block_v2(cases: list[dict], labels_tbl: pd.DataFrame, labels: list[str]) -> str:
    w = {l: 0.0 for l in labels}
    for c in cases:
        if c.get("label") in w:
            w[c["label"]] += max(0.0, float(c["score"]))
    tot = sum(w.values()) or 1.0
    dist = ", ".join(f"{l} {100 * w[l] / tot:.0f}%" for l in labels)
    lines = [f"Evidence from the {len(cases)} most acoustically similar cases (other patients, labels known):",
             f"Similarity-weighted label distribution: {dist}.",
             "These labels are informative but not perfect. Start from this distribution and move away from it only "
             "where the clip itself gives clear acoustic evidence. Individual cases:"]
    for rank, c in enumerate(cases, 1):
        row = labels_tbl.loc[c["uid"]]
        meta = [f"{name}={row[col]}" for col, name in CASE_META.get(row["corpus"], [])
                if col in row and pd.notna(row[col]) and str(row[col]) not in ("", "nan")]
        lab = c.get("label") if c.get("label") is not None else "unknown"
        lines.append(f"[{rank}] similarity={c['score']:.3f}; label={lab}" + ("; " + "; ".join(meta) if meta else ""))
    return "\n".join(lines)


def render_v2(task: str, labels: list[str], condition: str, cases: list[dict] | None = None,
              labels_tbl: pd.DataFrame | None = None) -> str:
    p = base_prompt_v2(task, labels)
    if condition == "rag":
        p += "\n\n" + evidence_block_v2(cases or [], labels_tbl, labels)
    return p


RENDERERS = {"v1": render, "v2": render_v2}


# ---------------------------------------------------------------------------------------------------
# Prompt v3: structured diagnostic REPORT (main evaluation). The model writes acoustic findings and an
# impression, then fills every report field; fields are scored against expert annotations
# (auscultrag/llm/report_fields.py). Both conditions share the same instructions; w/ RAG adds the
# retrieved cases together with THEIR full expert annotations (audio-to-audio retrieval supplies the
# sound-level evidence the model is asked to reason over).
# ---------------------------------------------------------------------------------------------------
REPORT_SUBJECT = {
    "circor_outcome": "a paediatric heart-sound (phonocardiogram) recording",
    "icbhi_chronic": "a chest auscultation recording of lung sounds",
    "icbhi_adventitious": "a chest auscultation recording of lung sounds",
    "sprsound_adventitious": "a paediatric chest auscultation recording of lung sounds",
    "kauh_adventitious": "a chest auscultation recording of lung sounds",
    "tbscreen_cough": "a short cough recording",
    "coswara_cough": "a voluntary cough recording",
    "coswara_respiratory": "a breathing-sound recording",
}


def base_prompt_v3(task: str, fields: dict) -> str:
    spec = "\n".join(f'  "{f}": one of [{", ".join(vals)}]  -- {desc}' for f, (vals, desc) in fields.items())
    keys = ", ".join(f'"{f}": "..."' for f in fields)
    return (
        "This is a research benchmark on clinical audio analysis; you are not giving medical advice.\n"
        f"An audio clip is attached (it may be short, about 1-15 seconds). It is {REPORT_SUBJECT[task]}.\n"
        "Write a short diagnostic report from the sound:\n"
        "Findings: 2-4 sentences describing what you hear (timing, character, adventitious sounds, rhythm).\n"
        "Impression: 1-2 sentences with your clinical interpretation.\n"
        "Then fill in every report field, choosing exactly one allowed value per field:\n"
        f"{spec}\n"
        "End your reply with ONE line of the form\n"
        f"REPORT: {{{keys}}}\n"
        "using valid JSON and values copied exactly from the allowed lists."
    )


def evidence_block_v3(cases: list[dict], case_fields: list[dict], fields: dict) -> str:
    primary = next(iter(fields))
    w: dict[str, float] = {}
    for c, cf in zip(cases, case_fields):
        v = cf.get(primary)
        if v is not None:
            w[v] = w.get(v, 0.0) + max(0.0, float(c["score"]))
    tot = sum(w.values()) or 1.0
    dist = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(w.items(), key=lambda kv: -kv[1]))
    lines = [f"Reference cases: the {len(cases)} recordings from OTHER patients that sound most similar to this clip "
             "(audio-to-audio retrieval), with their expert annotations.",
             f"Similarity-weighted distribution of '{primary}' among them: {dist}.",
             "These annotations are informative but not perfect: use them as evidence, and depart from them where the "
             "clip itself clearly sounds different."]
    for rank, (c, cf) in enumerate(zip(cases, case_fields), 1):
        ann = "; ".join(f"{k}={v}" for k, v in cf.items() if v is not None)
        lines.append(f"[{rank}] similarity={c['score']:.3f}; {ann}")
    return "\n".join(lines)


def render_v3(task: str, fields: dict, condition: str, cases: list[dict] | None = None,
              case_fields: list[dict] | None = None) -> str:
    p = base_prompt_v3(task, fields)
    if condition == "rag":
        p += "\n\n" + evidence_block_v3(cases or [], case_fields or [], fields)
    return p
