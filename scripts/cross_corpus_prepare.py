"""Stage: cross_corpus_prepare. Tier: cpu. Distribution-shift study (R2: "value beyond in-distribution
nearest-neighbour label transfer").

The three lung-sound corpora (ICBHI, SPRSound, KAUH) share a harmonised annotation: adventitious
(Normal/CAS/DAS/CAS&DAS; KAUH has no CAS&DAS), crackles, wheezes. For the frozen test queries of the three
adventitious tasks, three conditions with the SAME prompt (harmonised fields only):
  norag@xc : audio only
  rag@in   : evidence from the query's own corpus (the main-run cases, harmonised fields)
  rag@xc   : evidence retrieved ONLY from the two OTHER corpora (different hospitals, devices, populations,
             age groups - SPRSound is paediatric - and label conventions); the prompt says so.
Retrieval for rag@xc uses the query task's frozen dev setting (N, m, k, centering on the new KB).
kNN on the same evidence is added by report_score.py (knn_rag@in, knn_rag@xc).

Usage: python scripts/cross_corpus_prepare.py --name report_test_v3_xcorpus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from auscultrag.llm.prompts import base_prompt_v3, evidence_block_v3  # noqa: E402
from auscultrag.llm.report_fields import FIELDS  # noqa: E402
from auscultrag.retrieval.engine import UnitIndex, retrieve_cases_multi  # noqa: E402
from auscultrag.retrieval.store import label_table, load_embeddings  # noqa: E402

TASKS = {"icbhi_adventitious": "icbhi", "sprsound_adventitious": "sprsound", "kauh_adventitious": "kauh"}
RENAME = {"wheeze": "wheezes"}                         # SPRSound spells it 'wheeze'
HARMONISED = ("adventitious", "crackles", "wheezes")
XC_NOTE = ("Note: this reference library was collected at OTHER hospitals with other devices and patient "
           "populations than this clip, and annotation conventions may differ; weigh the evidence accordingly.")


def harm(d: dict) -> dict:
    d = {RENAME.get(k, k): v for k, v in d.items()}
    return {k: d.get(k) for k in HARMONISED}


def fields_for(task: str) -> dict:
    f = {RENAME.get(k, k): v for k, v in FIELDS[task].items()}
    return {k: f[k] for k in HARMONISED}


def _norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ARTIFACT_ROOT / "llm" / "report_test_v3" / "queue.jsonl")
    ap.add_argument("--name", default="report_test_v3_xcorpus")
    a = ap.parse_args()
    sel = json.loads((ARTIFACT_ROOT / "selection" / "retrieval_selection_dev.json").read_text())
    src = [json.loads(l) for l in a.src.open()]
    rag = [it for it in src if it["condition"] == "rag" and it["task"] in TASKS]
    emb, units = load_embeddings("clap")
    labels = label_table()
    rec_units = units.groupby("uid").indices
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {u: harm(dict(zip(g.field, g.value))) for (t, u), g in gt.groupby(["task", "uid"], sort=False) if t in TASKS}
    corpus_of_unit = units.corpus.to_numpy()
    labelled = units.fold.notna().to_numpy() & (units.family.to_numpy() == "respiratory")
    out = ARTIFACT_ROOT / "llm" / a.name / "queue.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    indexes = {}
    n_same = []
    with out.open("w") as f:
        for it in rag:
            task, corpus, s = it["task"], TASKS[it["task"]], sel[it["task"]]
            if task not in indexes:
                mask = labelled & np.isin(corpus_of_unit, [c for c in TASKS.values() if c != corpus])
                mask &= np.array([u in gt_map for u in units.uid])          # cases must carry an annotation
                kb = emb[mask]
                mu = kb.mean(axis=0, keepdims=True) if s["center"] else None
                indexes[task] = (UnitIndex(_norm(kb - mu) if s["center"] else kb, units[mask]), mu)
            index, mu = indexes[task]
            qe = emb[rec_units[it["uid"]]]
            if s["center"]:
                qe = _norm(qe - mu)
            row = labels.loc[it["uid"]]
            query = {"uid": it["uid"], "patient_id": it["patient_id"], "dup_group_id": row.dup_group_id}
            xc, _ = retrieve_cases_multi(index, qe, query, [s["top_n"]], [s["m"]], s["k"])[(s["top_n"], s["m"])]
            n_same.append(np.mean([c["corpus"] == corpus for c in xc]) if xc else np.nan)
            fields = fields_for(task)
            base = base_prompt_v3(task, fields)
            in_cases = [{"score": sc} for sc in it["case_scores"]]
            in_cf = [harm(cf) for cf in it["case_fields"]]
            xc_cf = [gt_map[c["uid"]] for c in xc]
            common = {**it, "fields": {k: list(v[0]) for k, v in fields.items()}, "gt_fields": harm(it["gt_fields"])}
            for cond, prompt, cf, sc, extra in (
                    ("norag@xc", base, [], [], {}),
                    ("rag@in", base + "\n\n" + evidence_block_v3(in_cases, in_cf, fields), in_cf, it["case_scores"], {}),
                    ("rag@xc", base + "\n\n" + evidence_block_v3(xc, xc_cf, fields) + "\n" + XC_NOTE, xc_cf,
                     [c["score"] for c in xc], {"case_uids": [c["uid"] for c in xc],
                                                "case_corpora": [c["corpus"] for c in xc]})):
                f.write(json.dumps({**common, "item_id": f"{task}|{cond}|{it['uid']}", "condition": cond,
                                    "prompt": prompt, "case_fields": cf, "case_scores": sc, **extra}) + "\n")
    print(f"wrote {out}: {3 * len(rag)} items; same-corpus rate in rag@xc evidence = {np.nanmean(n_same):.3f} (must be 0)")


if __name__ == "__main__":
    main()
