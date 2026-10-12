"""Stage: text_retrieval_prepare. Tier: cpu/small_gpu (CLAP text tower). Text-retrieval baseline (R1-W2: "is
audio-to-audio retrieval better than text-based retrieval?").

The query is DESCRIBED in words first: the Findings + Impression an audio-LLM wrote without evidence
(gpt-audio-1.5, condition norag of the main run). That text is used to retrieve cases in two ways:
  t2a : CLAP text embedding -> the SAME audio KB units (cross-modal; same KB, same patient-diverse case
        aggregation as AuscultRAG; only the query modality changes from audio to its verbal description)
  t2t : CLAP text embedding -> text of each KB recording's expert annotation ("crackles: yes; ...")
The retrieved cases are rendered in the SAME rag prompt (audio attached), so rag@t2a / rag@t2t vs rag differ
only in how the evidence was found. kNN on the same evidence is added by report_score.py.

Usage: python scripts/text_retrieval_prepare.py --describer gpt-audio-1.5 --name report_test_v3_textret
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auscultrag.config import ARTIFACT_ROOT, RAW_ROOT  # noqa: E402
from auscultrag.llm.prompts import render_v3  # noqa: E402
from auscultrag.llm.report_fields import FIELDS  # noqa: E402
from auscultrag.retrieval.engine import UnitIndex, retrieve_cases_multi  # noqa: E402
from auscultrag.retrieval.store import kb_mask, label_table, load_embeddings, load_tasks  # noqa: E402
from embed_units import MODEL_ID, REVISION  # noqa: E402


def description(text: str | None) -> str:
    t = re.split(r"REPORT\s*[:：]", text or "", flags=re.I)[0]
    return re.sub(r"\s+", " ", t).strip()[:1500] or "no description"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ARTIFACT_ROOT / "llm" / "report_test_v3" / "queue.jsonl")
    ap.add_argument("--describer", default="gpt-audio-1.5")
    ap.add_argument("--name", default="report_test_v3_textret")
    a = ap.parse_args()
    sel = json.loads((ARTIFACT_ROOT / "selection" / "retrieval_selection_dev.json").read_text())
    src = [json.loads(l) for l in a.src.open()]
    resp = {}
    for l in a.src.with_name(f"responses_{a.describer}.jsonl").open():
        r = json.loads(l)
        if r.get("error") is None:
            resp[r["item_id"]] = r["text"]
    rag = [it for it in src if it["condition"] == "rag"]
    desc = {it["uid"]: description(resp.get(f"{it['task']}|norag|{it['uid']}")) for it in rag}

    from transformers import ClapModel, ClapProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=RAW_ROOT / "hf_models").eval().to(dev)
    proc = ClapProcessor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=RAW_ROOT / "hf_models")

    def text_emb(texts: list[str]) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                inp = proc(text=texts[i: i + 128], return_tensors="pt", padding=True, truncation=True)
                e = model.get_text_features(**{k: v.to(dev) for k, v in inp.items()})
                out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.vstack(out)

    tasks = load_tasks()
    labels = label_table()
    emb, units = load_embeddings("clap")
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {(t, u): dict(zip(g.field, g.value)) for (t, u), g in gt.groupby(["task", "uid"], sort=False)}
    qtext = dict(zip(desc, text_emb(list(desc.values()))))
    out = ARTIFACT_ROOT / "llm" / a.name / "queue.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    with out.open("w") as f:
        for it in rag:
            t, s, r = it["task"], sel[it["task"]], it["round"]
            if (t, r) not in cache:
                td = tasks[t]
                mask = kb_mask(units, td["family"], r, td["corpus"], "corpus", False)
                kb_units = units[mask]
                recs = [u for u in kb_units.uid.unique() if (t, u) in gt_map]
                ann = [f"{td['family']} sound. " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in gt_map[(t, u)].items()
                                                              if v is not None) for u in recs]
                cache[(t, r)] = (UnitIndex(emb[mask], kb_units), recs, text_emb(ann),
                                 labels.loc[recs, "patient_id"].to_numpy())
            index, recs, ann_emb, rec_pid = cache[(t, r)]
            q = qtext[it["uid"]][None]
            row = labels.loc[it["uid"]]
            query = {"uid": it["uid"], "patient_id": it["patient_id"], "dup_group_id": row.dup_group_id}
            t2a, _ = retrieve_cases_multi(index, q, query, [s["top_n"]], [1], s["k"])[(s["top_n"], 1)]
            sims = ann_emb @ q[0]
            t2t, seen = [], {it["patient_id"]}
            for j in np.argsort(-sims):
                if rec_pid[j] in seen:
                    continue
                seen.add(rec_pid[j])
                t2t.append({"uid": recs[j], "patient_id": rec_pid[j], "score": float(sims[j])})
                if len(t2t) == s["k"]:
                    break
            for cond, cases in (("rag@t2a", t2a), ("rag@t2t", t2t)):
                cf = [gt_map.get((t, c["uid"]), {}) for c in cases]
                # displayed similarities = the audio run's values (as in make_control_queue.py), so the prompts
                # differ only in WHICH cases were found; the text-retrieval scores are kept as text_scores
                shown = [{**c, "score": sc} for c, sc in zip(cases, it["case_scores"])]
                f.write(json.dumps({**it, "item_id": f"{t}|{cond}|{it['uid']}", "condition": cond,
                                    "query_description": desc[it["uid"]],
                                    "prompt": render_v3(t, FIELDS[t], "rag", shown, cf), "case_fields": cf,
                                    "case_scores": it["case_scores"][: len(cases)],
                                    "text_scores": [c["score"] for c in cases],
                                    "case_uids": [c["uid"] for c in cases]}) + "\n")
    print(f"wrote {out}: {2 * len(rag)} items")


if __name__ == "__main__":
    main()
