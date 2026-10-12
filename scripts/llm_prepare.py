"""Stage: llm_prepare. Tier: cpu. Materialises the LLM work queue (R12) so the API/GPU stage only sends.

For each selected task, samples query recordings from the requested round, writes the query audio as
16 kHz mono 16-bit wav (cropped to MAX_AUDIO_SEC, starting at the first unit), renders the prompt for
each condition (norag / rag), and writes queue.jsonl. Retrieval results must already exist
(scripts/run_retrieval.py) for the given retrieval config.

Sampling (pilot): per task, up to --per-task queries, ONE recording per patient, class-balanced where
possible (equal share per class, remainder filled at random), seeded.
Usage: python scripts/llm_prepare.py --round 0 --per-task 100 --retrieval-cfg scope-corpus_cv-0_N200_m3_k10 --k 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, LABEL_SPACES, SEED  # noqa: E402
from auscultrag.llm.prompts import RENDERERS  # noqa: E402
from auscultrag.retrieval.store import label_table, load_tasks  # noqa: E402

MAX_AUDIO_SEC = 15.0
LLM_SR = 16000


def write_query_audio(src: str, start: float, out: Path, min_sec: float = 0.0) -> float:
    info = sf.info(src)
    a = int(start * info.samplerate)
    b = a + int(MAX_AUDIO_SEC * info.samplerate)
    x, sr = sf.read(src, start=a, stop=min(b, info.frames), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    g = np.gcd(sr, LLM_SR)
    x = resample_poly(x, LLM_SR // g, sr // g)
    peak = np.max(np.abs(x)) or 1.0
    x = 0.9 * x / peak
    short = int(min_sec * LLM_SR) - len(x)
    if short > 0:  # centre short clips in silence so the model registers the audio (LLM input only)
        x = np.pad(x, (short // 2, short - short // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, x, LLM_SR, subtype="PCM_16")
    return len(x) / LLM_SR


def sample(results: list[dict], per_task: int, rng: random.Random) -> list[dict]:
    by_patient = {}
    for r in sorted(results, key=lambda r: r["uid"]):
        by_patient.setdefault(r["patient_id"], []).append(r)
    one = [rng.choice(v) for _, v in sorted(by_patient.items())]
    by_class = {}
    for r in one:
        by_class.setdefault(r["y_true"], []).append(r)
    for v in by_class.values():
        rng.shuffle(v)
    share = max(1, per_task // max(1, len(by_class)))
    picked = [r for v in by_class.values() for r in v[:share]]
    rest = [r for v in by_class.values() for r in v[share:]]
    rng.shuffle(rest)
    picked += rest[: max(0, per_task - len(picked))]
    return sorted(picked, key=lambda r: r["uid"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--per-task", type=int, default=100)
    ap.add_argument("--retrieval-cfg", default=None, help="one retrieval config for all tasks")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--selection", type=Path, default=None,
                    help="per-task retrieval config + k chosen on dev (select_retrieval_config.py)")
    ap.add_argument("--encoder", default="clap")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--name", default=None)
    ap.add_argument("--prompt-version", choices=sorted(RENDERERS), default="v1")
    ap.add_argument("--min-audio-sec", type=float, default=0.0, help="pad shorter query clips with silence (LLM input only)")
    a = ap.parse_args()
    if (a.selection is None) == (a.retrieval_cfg is None):
        sys.exit("give exactly one of --selection or --retrieval-cfg")
    selection = json.loads(a.selection.read_text()) if a.selection else None
    name = a.name or (f"round{a.round}_n{a.per_task}_dev-selected" if selection else
                      f"round{a.round}_n{a.per_task}_{a.retrieval_cfg}_k{a.k}")
    out_dir = ARTIFACT_ROOT / "llm" / name
    labels_tbl = label_table()
    units = pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet", columns=["uid", "start_sec"])
    first_start = units.groupby("uid").start_sec.min()
    tasks = load_tasks()
    if a.tasks != "all":
        tasks = {t: tasks[t] for t in a.tasks.split(",")}
    rng = random.Random(SEED)
    queue = []
    for tname, t in tasks.items():
        cfg, k = (selection[tname]["retrieval_cfg"], selection[tname]["k"]) if selection else (a.retrieval_cfg, a.k)
        rpath = ARTIFACT_ROOT / "retrieval" / a.encoder / cfg / tname / f"round{a.round}.jsonl"
        results = [json.loads(line) for line in rpath.open()]
        space = list(LABEL_SPACES[t["label_space"]])
        for r in sample(results, a.per_task, rng):
            row = labels_tbl.loc[r["uid"]]
            wav = out_dir / "audio" / f"{r['uid'].replace(':', '__').replace('/', '_')}.wav"
            dur = write_query_audio(row.filepath, float(first_start.get(r["uid"], 0.0)), wav, a.min_audio_sec) if not wav.exists() \
                else sf.info(wav).duration
            for cond in ("norag", "rag"):
                queue.append({
                    "item_id": f"{tname}|{cond}|{r['uid']}", "task": tname, "condition": cond, "uid": r["uid"],
                    "patient_id": r["patient_id"], "y_true": r["y_true"], "labels": space,
                    "audio_path": str(wav), "audio_sec": round(dur, 3),
                    "prompt": RENDERERS[a.prompt_version](tname, space, cond, r["cases"][:k], labels_tbl),
                    "prompt_version": a.prompt_version,
                    "case_labels": [c.get("label") for c in r["cases"][:k]],
                    "case_scores": [c["score"] for c in r["cases"][:k]],
                    "retrieval_cfg": cfg, "k": k,
                })
        print(f"{tname}: {sum(1 for q in queue if q['task'] == tname) // 2} queries", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "queue.jsonl").open("w") as f:
        for q in queue:
            f.write(json.dumps(q) + "\n")
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "selection": selection,
                                                     "max_audio_sec": MAX_AUDIO_SEC, "llm_sr": LLM_SR}, indent=2))
    print(f"wrote {out_dir / 'queue.jsonl'}: {len(queue)} items")


if __name__ == "__main__":
    main()
