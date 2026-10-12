"""Stage: report_prepare. Tier: cpu. Work queue for the structured-report evaluation (prompt v3).

Queries are sampled from the given rounds (0 = dev; 1,2,3,4 = the four test rounds pooled), ONE recording
per patient, class-balanced on the task's primary label, seeded. Each query is rendered for both
conditions (norag / rag). Retrieval settings come from the dev selection; retrieved cases carry their
expert report fields (ground truth table), which are also stored for the per-field kNN baseline.

Usage:
  python scripts/report_prepare.py --rounds 0 --per-task 30 --name report_dev_v3
  python scripts/report_prepare.py --rounds 1,2,3,4 --per-task 200 --name report_test_v3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auscultrag.config import ARTIFACT_ROOT, SEED  # noqa: E402
from auscultrag.llm.prompts import render_v3  # noqa: E402
from auscultrag.llm.report_fields import FIELDS  # noqa: E402
from auscultrag.retrieval.store import label_table, load_tasks  # noqa: E402
from llm_prepare import sample, write_query_audio  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--per-task", type=int, default=200)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--min-audio-sec", type=float, default=2.0)
    ap.add_argument("--selection", type=Path, default=ARTIFACT_ROOT / "selection" / "retrieval_selection_dev.json")
    a = ap.parse_args()
    rounds = [int(r) for r in a.rounds.split(",")]
    if 0 in rounds and len(rounds) > 1:
        sys.exit("never mix the dev round with test rounds")
    selection = json.loads(a.selection.read_text())
    out_dir = ARTIFACT_ROOT / "llm" / a.name
    labels_tbl = label_table()
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {(t, u): dict(zip(g.field, g.value)) for (t, u), g in gt.groupby(["task", "uid"], sort=False)}
    units = pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet", columns=["uid", "start_sec"])
    first_start = units.groupby("uid").start_sec.min()
    tasks = load_tasks()
    if a.tasks != "all":
        tasks = {t: tasks[t] for t in a.tasks.split(",")}
    rng = random.Random(SEED)
    queue = []
    for tname in tasks:
        s = selection[tname]
        results = []
        for r in rounds:
            p = ARTIFACT_ROOT / "retrieval" / "clap" / s["retrieval_cfg"] / tname / f"round{r}.jsonl"
            results += [json.loads(line) | {"round": r} for line in p.open()]
        fields = FIELDS[tname]
        for r in sample(results, a.per_task, rng):
            row = labels_tbl.loc[r["uid"]]
            wav = out_dir / "audio" / f"{r['uid'].replace(':', '__').replace('/', '_')}.wav"
            dur = (write_query_audio(row.filepath, float(first_start.get(r["uid"], 0.0)), wav, a.min_audio_sec)
                   if not wav.exists() else sf.info(wav).duration)
            cases = r["cases"][: s["k"]]
            case_fields = [gt_map.get((tname, c["uid"]), {}) for c in cases]
            for cond in ("norag", "rag"):
                queue.append({
                    "item_id": f"{tname}|{cond}|{r['uid']}", "task": tname, "condition": cond, "uid": r["uid"],
                    "patient_id": r["patient_id"], "round": r["round"], "y_true": r["y_true"],
                    "audio_path": str(wav), "audio_sec": round(dur, 3), "prompt_version": "v3",
                    "prompt": render_v3(tname, fields, cond, cases, case_fields),
                    "fields": {f: list(v[0]) for f, v in fields.items()},
                    "gt_fields": gt_map.get((tname, r["uid"]), {}),
                    "case_fields": case_fields, "case_scores": [c["score"] for c in cases],
                    "retrieval_cfg": s["retrieval_cfg"], "k": s["k"],
                })
        print(f"{tname}: {sum(1 for q in queue if q['task'] == tname) // 2} queries", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "queue.jsonl").open("w") as f:
        for q in queue:
            f.write(json.dumps(q) + "\n")
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "selection": selection}, indent=2))
    print(f"wrote {out_dir / 'queue.jsonl'}: {len(queue)} items")


if __name__ == "__main__":
    main()
