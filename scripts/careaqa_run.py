"""Stage: careaqa_run. Tier: big_gpu (envs/gpu). CaReAQA on the frozen test queries (cardiac + lung tasks only;
CaReAQA was not trained on cough). CaReAQA is a short-answer QA model, so each report field it can address is
asked with its OWN training phrasing (most frequent CaReSound question for that dataset/field); answers are
mapped to field values with fixed keyword rules (below), written as a REPORT line so report_score.py scores
them unchanged. Fields without a native question are left empty; compare on the supported fields
(SUPPORTED) with every other model (aggregate_results.py, careaqa table). Protocol calibrated on CaReSound's
own test split: LoRA alpha 32, layout "question: Q" [audio prefix] " answer" (scripts/careaqa.py).

Usage: $GPY scripts/careaqa_run.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from careaqa import CaReAQA  # noqa: E402

ALPHA, LAYOUT = 32.0, "q_p_answer"
Q = {
    "murmur": "Were any murmurs detected during the cardiac auscultation?",
    "outcome": "What is the overall outcome of the cardiac auscultation?",
    "icbhi_diag": "What is the diagnosis based on the auscultation findings?",
    "crackles": "Are crackles present in the auscultation findings?",
    "wheezes": "Are wheezes present in the auscultation findings?",
    "spr_abnormal": "Were any abnormal lung sounds detected during auscultation?",
    "spr_crackles": "Were any crackles heard during auscultation?",
    "spr_wheeze": "Did the auscultation findings indicate the presence of wheezes?",
    "kauh_diag": "What is the diagnosis?",
    "kauh_sound": "What type of sound is heard?",
}
TASK_Q = {"circor_outcome": ["murmur", "outcome"], "icbhi_chronic": ["icbhi_diag", "crackles", "wheezes"],
          "icbhi_adventitious": ["icbhi_diag", "crackles", "wheezes"],
          "sprsound_adventitious": ["spr_abnormal", "spr_crackles", "spr_wheeze"],
          "kauh_adventitious": ["kauh_diag", "kauh_sound"]}
SUPPORTED = {"circor_outcome": ["outcome", "murmur"], "icbhi_chronic": ["condition", "diagnosis", "crackles", "wheezes"],
             "icbhi_adventitious": ["adventitious", "crackles", "wheezes", "condition"],
             "sprsound_adventitious": ["adventitious", "crackles", "wheeze"],
             "kauh_adventitious": ["adventitious", "wheezes", "crackles", "diagnosis"]}
ICBHI_DIAG = [("COPD", r"\bCOPD\b|chronic obstructive"), ("Bronchiectasis", r"bronchiectasis"), ("Asthma", r"asthma"),
              ("URTI", r"\bURTI\b|upper respiratory"), ("LRTI", r"\bLRTI\b|lower respiratory"),
              ("Pneumonia", r"pneumonia"), ("Bronchiolitis", r"bronchiolitis"), ("Healthy", r"healthy|normal")]
CHRONIC = {"COPD": "Chronic", "Bronchiectasis": "Chronic", "Asthma": "Chronic", "URTI": "NonChronic", "LRTI": "NonChronic",
           "Pneumonia": "NonChronic", "Bronchiolitis": "NonChronic", "Healthy": "Healthy"}
KAUH_DIAG = [("Asthma", r"asthma"), ("COPD", r"\bCOPD\b|chronic obstructive"), ("Heart failure", r"heart failure"),
             ("Pneumonia", r"pneumonia"), ("Lung fibrosis", r"fibrosis"), ("Normal", r"normal|healthy")]


def yes(a: str) -> str | None:
    a = a.strip().lower()
    if re.match(r"(no\b|none|there (were|was|are|is) no|not\b)", a) or re.search(r"\b(not|no|absent|absence)\b", a[:40]):
        return "no"
    if re.match(r"yes\b", a) or re.search(r"\b(present|detected|heard|were|was)\b", a[:40]):
        return "yes"
    return None


def first(rules, a):
    return next((v for v, r in rules if re.search(r, a, re.I)), None)


def adv(crackles, wheezes):
    if crackles is None or wheezes is None:
        return None
    return {("yes", "yes"): "CAS&DAS", ("yes", "no"): "DAS", ("no", "yes"): "CAS", ("no", "no"): "Normal"}[(crackles, wheezes)]


def to_fields(task: str, ans: dict) -> dict:
    if task == "circor_outcome":
        m = yes(ans["murmur"])
        o = "Abnormal" if re.search(r"abnormal", ans["outcome"], re.I) else ("Normal" if re.search(r"normal", ans["outcome"], re.I) else None)
        return {"outcome": o, "murmur": {"yes": "Present", "no": "Absent"}.get(m)}
    if task in ("icbhi_chronic", "icbhi_adventitious"):
        d = first(ICBHI_DIAG, ans["icbhi_diag"])
        c, w = yes(ans["crackles"]), yes(ans["wheezes"])
        f = {"diagnosis": d, "condition": CHRONIC.get(d), "crackles": c, "wheezes": w}
        if task == "icbhi_adventitious":
            f = {"adventitious": adv(c, w), "crackles": c, "wheezes": w, "condition": CHRONIC.get(d)}
        return f
    if task == "sprsound_adventitious":
        ab, c, w = yes(ans["spr_abnormal"]), yes(ans["spr_crackles"]), yes(ans["spr_wheeze"])
        a = "Normal" if ab == "no" else adv(c, w)
        if a == "Normal" and ab == "yes":
            a = None
        return {"adventitious": a, "crackles": c, "wheeze": w}
    if task == "kauh_adventitious":
        s = ans["kauh_sound"].lower()
        w = "yes" if "wheez" in s else "no"
        c = "yes" if re.search(r"crep|crackl", s) else "no"
        a = "Normal" if re.search(r"normal", s) and w == c == "no" else {("no", "yes"): "CAS", ("yes", "no"): "DAS"}.get((c, w))
        return {"adventitious": a, "wheezes": w, "crackles": c, "diagnosis": first(KAUH_DIAG, ans["kauh_diag"])}
    raise ValueError(task)


def main() -> None:
    src = ARTIFACT_ROOT / "llm" / "report_test_v3" / "queue.jsonl"
    out_dir = ARTIFACT_ROOT / "llm" / "report_test_v3_careaqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in src.open()]
    items = [it for it in items if it["condition"] == "norag" and it["task"] in TASK_Q]
    with (out_dir / "queue.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    m = CaReAQA(ALPHA)
    with (out_dir / "responses_careaqa.jsonl").open("w") as f:
        for i, it in enumerate(items):
            ans = {k: m.answer(it["audio_path"], Q[k], LAYOUT) for k in TASK_Q[it["task"]]}
            fields = {k: v for k, v in to_fields(it["task"], ans).items() if v is not None}
            f.write(json.dumps({"item_id": it["item_id"], "model": "careaqa", "response_model": "tsnngw/CaReAQA",
                                "text": "Answers: " + json.dumps(ans) + "\nREPORT: " + json.dumps(fields),
                                "raw_answers": ans, "error": None}) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(items)}", ans, fields, flush=True)
    print("done")


if __name__ == "__main__":
    main()
