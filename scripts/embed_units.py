"""Stage: embed_units. Tier: small_gpu (runs on CPU when no GPU is present; fits in 12 GB).

Encodes every unit in units.parquet with CLAP (laion/clap-htsat-unfused, pinned revision) into a
unit-norm 512-d embedding.

Signal contract (R9, cross-corpus comparability):
  * each unit is band-limited to a common per-family bandwidth by resampling to FAMILY_SR
    (cardiac/respiratory 4 kHz, cough 8 kHz) before upsampling to CLAP's 48 kHz, so sample rate /
    bandwidth cannot identify the corpus;
  * short units are repeat-padded to CLAP's 10 s window (never zero-padded), so TBScreen's zero
    padding is not a cue.

Resumable (R11): embeddings are written in shards (unit_ids + float32 matrix); on start, units in
existing shards are skipped; a kill loses at most one shard in progress.
Output: $AUSCULTRAG_ARTIFACTS/embeddings/clap/shard_*.npz  (+ config.json)
Usage: python scripts/embed_units.py [--batch 32] [--shard 2048] [--device auto]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, RAW_ROOT  # noqa: E402
from auscultrag.stage import config_hash, git_commit  # noqa: E402

MODEL_ID = "laion/clap-htsat-unfused"
REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
CLAP_SR = 48000
FAMILY_SR = {"cardiac": 4000, "respiratory": 4000, "cough": 8000}


def load_unit(path: str, start: float, end: float, family: str) -> np.ndarray:
    info = sf.info(path)
    a, b = int(start * info.samplerate), int(end * info.samplerate)
    x, sr = sf.read(path, start=a, stop=max(b, a + 1), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    target = FAMILY_SR[family]
    g = np.gcd(sr, target)
    x = resample_poly(x, target // g, sr // g)          # band-limit to the family's common bandwidth
    x = resample_poly(x, CLAP_SR // target, 1)          # back up to CLAP's rate
    return x.astype(np.float32)


class Units(Dataset):
    def __init__(self, df: pd.DataFrame, paths: dict[str, str]):
        self.rows = df[["unit_id", "uid", "start_sec", "end_sec", "family"]].to_dict("records")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            return r["unit_id"], load_unit(self.paths[r["uid"]], r["start_sec"], r["end_sec"], r["family"])
        except Exception as exc:
            return r["unit_id"], f"ERROR {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if device == "cpu":
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) - a.workers))

    from transformers import ClapModel, ClapProcessor
    cache = RAW_ROOT / "hf_models"
    model = ClapModel.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=cache).eval().to(device)
    proc = ClapProcessor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=cache)

    units = pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet")
    manifest = pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest.parquet", columns=["uid", "filepath"])
    paths = dict(zip(manifest.uid, manifest.filepath))
    out_dir = ARTIFACT_ROOT / "embeddings" / "clap"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL_ID, "revision": REVISION, "family_sr": FAMILY_SR, "clap_sr": CLAP_SR,
              "padding": "repeatpad", "units_meta": json.loads((ARTIFACT_ROOT / "units" / "units.parquet.meta.json").read_text())["config_hash"]}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text())["config_hash"] != config_hash(config):
        sys.exit(f"{out_dir} holds embeddings from a different config; move it to trash/ before re-running")
    cfg_path.write_text(json.dumps({**config, "config_hash": config_hash(config), "git_commit": git_commit(),
                                    "device": device}, indent=2))

    done = set()
    shards = sorted(out_dir.glob("shard_*.npz"))
    for s in shards:
        done.update(np.load(s, allow_pickle=False)["unit_ids"].tolist())
    todo = units[~units.unit_id.isin(done)]
    print(f"{len(units)} units, {len(done)} already embedded, {len(todo)} to go on {device}", flush=True)
    loader = DataLoader(Units(todo, paths), batch_size=a.batch, num_workers=a.workers,
                        collate_fn=lambda b: b, prefetch_factor=4 if a.workers else None)
    next_shard = len(shards)
    buf_ids, buf_emb, errors = [], [], []
    t0, n = time.time(), 0

    def flush() -> None:
        nonlocal next_shard, buf_ids, buf_emb
        if not buf_ids:
            return
        path = out_dir / f"shard_{next_shard:05d}.npz"
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, unit_ids=np.array(buf_ids), emb=np.vstack(buf_emb).astype(np.float32))
        tmp.replace(path)
        next_shard += 1
        buf_ids, buf_emb = [], []

    with torch.inference_mode():
        for batch in loader:
            ok = [(u, x) for u, x in batch if not isinstance(x, str)]
            errors += [(u, x) for u, x in batch if isinstance(x, str)]
            if ok:
                inp = proc(audios=[x for _, x in ok], sampling_rate=CLAP_SR, return_tensors="pt", padding="repeatpad")
                emb = model.get_audio_features(**{k: v.to(device) for k, v in inp.items()})
                emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
                buf_ids += [u for u, _ in ok]
                buf_emb.append(emb)
            n += len(batch)
            if len(buf_ids) >= a.shard:
                flush()
                rate = n / (time.time() - t0)
                print(f"embedded {n}/{len(todo)}  {rate:.1f} units/s  ETA {(len(todo) - n) / rate / 60:.0f} min", flush=True)
    flush()
    if errors:
        with open(out_dir / "errors.jsonl", "a") as f:
            for u, e in errors:
                f.write(json.dumps({"unit_id": u, "error": e}) + "\n")
    print(f"done: {n} processed, {len(errors)} errors")


if __name__ == "__main__":
    main()
