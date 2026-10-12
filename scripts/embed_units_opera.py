"""Stage: embed_units_opera. Tier: small_gpu (runs on CPU when no GPU). OPERA encoders (Zhang et al.,
NeurIPS 2024 D&B; github.com/evelyn0414/OPERA @ 3622310e, weights hf: evelyn0414/OPERA).

Same units and the same signal contract as embed_units.py (CLAP), so encoders are compared fairly:
each unit is band-limited to the family bandwidth (FAMILY_SR), resampled to OPERA's 16 kHz, repeat-padded
to 8 s (OPERA's own 'repeat' padding), converted with OPERA's mel front-end (pre_process_audio_mel_t,
f_max = 8 kHz as in get_entire_signal_librosa), and encoded; embeddings are L2-normalised.
Resumable shards like embed_units.py.

Usage: python scripts/embed_units_opera.py --encoder operaCT [--batch 32]
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

OPERA_DIR = RAW_ROOT / "external" / "OPERA"
OPERA_COMMIT = "3622310e667afb8aa40169050b4dd45de75946a2"
sys.path.insert(0, str(OPERA_DIR))
FAMILY_SR = {"cardiac": 4000, "respiratory": 4000, "cough": 8000}
SR, INPUT_SEC = 16000, 8
DIMS = {"operaCT": 768, "operaCE": 1280}


def load_unit(path: str, start: float, end: float, family: str) -> np.ndarray:
    from src.util import pre_process_audio_mel_t
    info = sf.info(path)
    a, b = int(start * info.samplerate), int(end * info.samplerate)
    x, sr = sf.read(path, start=a, stop=max(b, a + 1), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    t = FAMILY_SR[family]
    g = np.gcd(sr, t)
    x = resample_poly(x, t // g, sr // g)
    x = resample_poly(x, SR // t, 1).astype(np.float32)
    need = SR * INPUT_SEC
    if len(x) == 0:
        x = np.zeros(1, dtype=np.float32)
    if len(x) < need:                                   # OPERA 'repeat' padding
        x = np.tile(x, int(np.ceil(need / len(x))))[:need]
    else:
        x = x[:need]
    return pre_process_audio_mel_t(x, sample_rate=SR, f_max=8000).astype(np.float32)


class Units(Dataset):
    def __init__(self, df, paths):
        self.rows = df[["unit_id", "uid", "start_sec", "end_sec", "family"]].to_dict("records")
        self.paths = paths

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            return r["unit_id"], load_unit(self.paths[r["uid"]], r["start_sec"], r["end_sec"], r["family"])
        except Exception as exc:
            return r["unit_id"], f"ERROR {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=sorted(DIMS), default="operaCT")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) - a.workers))
    from huggingface_hub import hf_hub_download
    from src.benchmark.model_util import initialize_pretrained_model
    ckpt_path = hf_hub_download("evelyn0414/OPERA", f"encoder-{a.encoder}.ckpt", cache_dir=RAW_ROOT / "hf_models")
    model = initialize_pretrained_model(a.encoder)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval().to(device)

    units = pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet")
    paths = dict(pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest.parquet", columns=["uid", "filepath"]).values)
    out_dir = ARTIFACT_ROOT / "embeddings" / a.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"encoder": a.encoder, "opera_commit": OPERA_COMMIT, "family_sr": FAMILY_SR, "sr": SR, "input_sec": INPUT_SEC,
              "padding": "repeat", "dim": DIMS[a.encoder],
              "units_meta": json.loads((ARTIFACT_ROOT / "units" / "units.parquet.meta.json").read_text())["config_hash"]}
    cfg = out_dir / "config.json"
    if cfg.exists() and json.loads(cfg.read_text())["config_hash"] != config_hash(config):
        sys.exit(f"{out_dir} holds embeddings from a different config; move it to trash/ first")
    cfg.write_text(json.dumps({**config, "config_hash": config_hash(config), "git_commit": git_commit(), "device": device,
                               "n_missing_keys": len(missing), "n_unexpected_keys": len(unexpected)}, indent=2))
    done = set()
    shards = sorted(out_dir.glob("shard_*.npz"))
    for s in shards:
        done.update(np.load(s)["unit_ids"].tolist())
    todo = units[~units.unit_id.isin(done)]
    print(f"{a.encoder}: {len(units)} units, {len(done)} done, {len(todo)} to go on {device}; "
          f"missing keys {len(missing)}, unexpected {len(unexpected)}", flush=True)
    loader = DataLoader(Units(todo, paths), batch_size=a.batch, num_workers=a.workers, collate_fn=lambda b: b,
                        prefetch_factor=4 if a.workers else None)
    next_shard, buf_ids, buf_emb, errors, n, t0 = len(shards), [], [], [], 0, time.time()

    def flush():
        nonlocal next_shard, buf_ids, buf_emb
        if buf_ids:
            p = out_dir / f"shard_{next_shard:05d}.npz"
            tmp = p.with_name(p.stem + ".tmp.npz")
            np.savez(tmp, unit_ids=np.array(buf_ids), emb=np.vstack(buf_emb).astype(np.float32))
            tmp.replace(p)
            next_shard += 1
            buf_ids, buf_emb = [], []

    with torch.inference_mode():
        for batch in loader:
            ok = [(u, x) for u, x in batch if not isinstance(x, str)]
            errors += [(u, x) for u, x in batch if isinstance(x, str)]
            if ok:
                x = torch.from_numpy(np.stack([m for _, m in ok])).to(device)
                e = model.extract_feature(x, DIMS[a.encoder])
                e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
                buf_ids += [u for u, _ in ok]
                buf_emb.append(e)
            n += len(batch)
            if len(buf_ids) >= a.shard:
                flush()
                r = n / (time.time() - t0)
                print(f"embedded {n}/{len(todo)}  {r:.1f} units/s  ETA {(len(todo) - n) / r / 60:.0f} min", flush=True)
    flush()
    if errors:
        with open(out_dir / "errors.jsonl", "a") as f:
            for u, e in errors:
                f.write(json.dumps({"unit_id": u, "error": e}) + "\n")
    print(f"done: {n} processed, {len(errors)} errors")


if __name__ == "__main__":
    main()
