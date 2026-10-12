"""Live tqdm view of a running stage, without touching the job itself (it only reads its outputs).

  python scripts/watch_progress.py embed      # CLAP embedding: counts units in finished shards
  python scripts/watch_progress.py llm --queue <dir>/queue.jsonl [--model gpt-audio-2025-08-28]
  python scripts/watch_progress.py grid [--log <file>] [--total 16]   # dev retrieval grid (one line per task x centering setting)

Ctrl-C to stop watching (the job keeps running).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402


def embed_counter(encoder: str):
    total = len(pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet", columns=["unit_id"]))
    d = ARTIFACT_ROOT / "embeddings" / encoder
    seen: dict[str, int] = {}

    def count() -> int:
        for s in d.glob("shard_*.npz"):
            if s.name not in seen:
                try:
                    seen[s.name] = len(np.load(s, allow_pickle=False)["unit_ids"])
                except Exception:  # shard being written right now
                    pass
        return sum(seen.values())
    return total, count, "units"


def llm_counter(queue: Path, model: str):
    total = sum(1 for _ in queue.open())
    out = queue.with_name(f"responses_{model}.jsonl")

    def count() -> int:
        if not out.exists():
            return 0
        ok = set()
        for line in out.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                ok.add(r["item_id"])
        return len(ok)
    return total, count, "calls"


def grid_counter(log: Path, total: int):
    def count() -> int:
        return sum(1 for line in log.open() if "best knn macroF1" in line) if log.exists() else 0
    return total, count, "tasks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["embed", "llm", "grid"])
    ap.add_argument("--log", type=Path, default=ARTIFACT_ROOT / "logs" / "retrieval_dev_grid.log")
    ap.add_argument("--total", type=int, default=16)
    ap.add_argument("--encoder", default="clap")
    ap.add_argument("--queue", type=Path)
    ap.add_argument("--model", default="gpt-audio-2025-08-28")
    ap.add_argument("--every", type=float, default=5.0, help="poll interval, seconds")
    a = ap.parse_args()
    if a.what == "embed":
        total, count, unit = embed_counter(a.encoder)
    elif a.what == "llm":
        total, count, unit = llm_counter(a.queue, a.model)
    else:
        total, count, unit = grid_counter(a.log, a.total)
    start = count()
    bar = tqdm(total=total, initial=start, unit=unit, dynamic_ncols=True, smoothing=0.1,
               desc=a.what)
    try:
        while True:
            n = count()
            if n > bar.n:
                bar.update(n - bar.n)
            if n >= total:
                break
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass
    bar.close()


if __name__ == "__main__":
    main()
