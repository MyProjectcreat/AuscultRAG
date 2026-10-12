"""Stage: make_control_queue. Tier: cpu. Evidence-relevance controls (R1-W5, AI-W2: "label-token priming").

For the SAME test queries as the main queue, the retrieved evidence is replaced while the prompt format,
the number of cases and the displayed similarity values stay identical (the query's own top-k scores are
reused), so only the acoustic relevance of the cases changes:
  rag_far    : ranks k+1..2k of the same retrieval (similar, but less similar neighbours)
  rag_random : k random knowledge-base cases of the same task (other folds, other patients, one per patient)
If AuscultRAG only primed the label vocabulary, these would score like the real evidence; a dose-response
real top-k > far > random shows that the gain comes from acoustic similarity.
The prompt states the cases are the most similar ones in all conditions (the model is blind to the swap).

Usage: python scripts/make_control_queue.py --src <dir>/queue.jsonl --name report_test_v3_controls
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, SEED  # noqa: E402
from auscultrag.llm.prompts import render_v3  # noqa: E402
from auscultrag.llm.report_fields import FIELDS  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    rng = random.Random(SEED + 1)
    src = [json.loads(l) for l in a.src.open()]
    rag = [it for it in src if it["condition"] == "rag"]
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {(t, u): dict(zip(g.field, g.value)) for (t, u), g in gt.groupby(["task", "uid"], sort=False)}

    # full retrieval results (top-20) and the KB pool per (task, round): every case retrieved for any query of
    # that round is a KB member of that round (KB = all other folds)
    results, pool = {}, defaultdict(dict)
    for task, cfg in {(it["task"], it["retrieval_cfg"]) for it in rag}:
        for r in (1, 2, 3, 4):
            for line in (ARTIFACT_ROOT / "retrieval" / "clap" / cfg / task / f"round{r}.jsonl").open():
                q = json.loads(line)
                results[(task, q["uid"])] = q
                for c in q["cases"]:
                    pool[(task, r)][c["uid"]] = c
    out = ARTIFACT_ROOT / "llm" / a.name / "queue.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = defaultdict(int)
    with out.open("w") as f:
        for it in rag:
            task, k = it["task"], it["k"]
            fields = FIELDS[task]
            top = results[(task, it["uid"])]["cases"]
            scores = [c["score"] for c in top[:k]]
            far = top[k: 2 * k]
            cands = [c for c in pool[(task, it["round"])].values() if c["patient_id"] != it["patient_id"]]
            rng.shuffle(cands)
            rand, seen = [], set()
            for c in cands:
                if c["patient_id"] not in seen:
                    rand.append(c)
                    seen.add(c["patient_id"])
                if len(rand) == k:
                    break
            for cond, cases in (("rag_far", far), ("rag_random", rand)):
                if len(cases) < k:
                    continue
                shown = [{**c, "score": s} for c, s in zip(cases, scores)]   # identical displayed similarities
                cf = [gt_map.get((task, c["uid"]), {}) for c in cases]
                f.write(json.dumps({**it, "item_id": f"{task}|{cond}|{it['uid']}", "condition": cond,
                                    "prompt": render_v3(task, fields, "rag", shown, cf),
                                    "case_fields": cf, "case_scores": scores,
                                    "control_case_uids": [c["uid"] for c in cases]}) + "\n")
                n[cond] += 1
    print(f"wrote {out}: {dict(n)}")


if __name__ == "__main__":
    main()
