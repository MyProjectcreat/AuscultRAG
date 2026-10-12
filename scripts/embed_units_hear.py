"""Stage: embed_units_hear. Tier: small_gpu (runs on CPU when no GPU). HeAR encoder (Baur et al., 2024,
arXiv:2403.02522; hf: google/hear-pytorch @ f791cd42, preprocessing github.com/Google-Health/hear @ 36bbb908).

Same units and the same signal contract as embed_units.py (CLAP) / embed_units_opera.py, so encoders are
compared fairly: each unit is band-limited to the family bandwidth (FAMILY_SR) and resampled to HeAR's 16 kHz.
HeAR encodes 2-s clips: a unit <= 2 s is repeat-padded to 2 s; a longer unit is cut into ceil(len / 2 s)
evenly spaced 2-s windows (at most MAX_WIN) whose pooled embeddings are averaged. Front-end is HeAR's own
preprocess_audio (mel-PCEN, 192x128); embedding = pooler_output (512-d), L2-normalised.
Resumable shards like embed_units.py. Needs HF_TOKEN (gated model) in .env.

Usage: python scripts/embed_units_hear.py [--batch 64]
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

HEAR_DIR = RAW_ROOT / "external" / "hear"
HEAR_COMMIT = "36bbb90879545a2f1106668048b1a12a61ff2a0d"
HF_REPO, HF_REV = "google/hear-pytorch", "f791cd42437c3e268c8ac84707e3508900f65f1a"
sys.path.insert(0, str(HEAR_DIR))
FAMILY_SR = {"cardiac": 4000, "respiratory": 4000, "cough": 8000}
SR, CLIP = 16000, 32000
MAX_WIN = 4
DIM = 512


def load_unit(path: str, start: float, end: float, family: str) -> np.ndarray:
    """Returns [n_win, 32000] float32 windows at 16 kHz."""
    info = sf.info(path)
    a, b = int(start * info.samplerate), int(end * info.samplerate)
    x, sr = sf.read(path, start=a, stop=max(b, a + 1), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    t = FAMILY_SR[family]
    g = np.gcd(sr, t)
    x = resample_poly(x, t // g, sr // g)
    x = resample_poly(x, SR // t, 1).astype(np.float32)
    if len(x) == 0:
        x = np.zeros(1, dtype=np.float32)
    if len(x) <= CLIP:                                  # repeat padding, as for CLAP / OPERA
        return np.tile(x, int(np.ceil(CLIP / len(x))))[None, :CLIP]
    n = min(MAX_WIN, int(np.ceil(len(x) / CLIP)))
    starts = np.linspace(0, len(x) - CLIP, n).astype(int)
    return np.stack([x[s:s + CLIP] for s in starts])


class HeAR(torch.nn.Module):
    """ViT-L encoder + HeAR's 512-d linear pooler on the CLS token. Equivalent to ViTModel's pooler with
    pooler_output_size=512, pooler_act=linear (transformers >= 4.50); built by hand to keep the pinned 4.49."""

    def __init__(self, vit, pooler):
        super().__init__()
        self.vit, self.pooler = vit, pooler

    def forward(self, x):
        return self.pooler(self.vit(x).last_hidden_state[:, 0])


def load_hear(device):
    from huggingface_hub import hf_hub_download
    from transformers import ViTConfig, ViTModel
    kw = {"revision": HF_REV, "cache_dir": RAW_ROOT / "hf_models"}
    cfg = ViTConfig.from_pretrained(HF_REPO, **kw)
    vit = ViTModel(cfg, add_pooling_layer=False)
    state = torch.load(hf_hub_download(HF_REPO, "pytorch_model.bin", **kw), map_location="cpu")
    pooler = torch.nn.Linear(cfg.hidden_size, DIM)
    pooler.load_state_dict({"weight": state.pop("pooler.dense.weight"), "bias": state.pop("pooler.dense.bias")})
    missing, unexpected = vit.load_state_dict({k.removeprefix("vit."): v for k, v in state.items()}, strict=False)
    if missing or unexpected:
        sys.exit(f"HeAR weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return HeAR(vit, pooler).eval().to(device)


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
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: embed only the first N pending units, no shards written")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) - a.workers))
    from python.data_processing.audio_utils import preprocess_audio
    model = load_hear(device)

    units = pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet")
    paths = dict(pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest.parquet", columns=["uid", "filepath"]).values)
    out_dir = ARTIFACT_ROOT / "embeddings" / "hear"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"encoder": "hear", "hf_repo": HF_REPO, "hf_revision": HF_REV, "hear_commit": HEAR_COMMIT,
              "family_sr": FAMILY_SR, "sr": SR, "clip_samples": CLIP, "max_windows": MAX_WIN, "padding": "repeat",
              "pool": "mean of window pooler_output", "dim": DIM,
              "units_meta": json.loads((ARTIFACT_ROOT / "units" / "units.parquet.meta.json").read_text())["config_hash"]}
    cfg = out_dir / "config.json"
    if not a.limit:
        if cfg.exists() and json.loads(cfg.read_text())["config_hash"] != config_hash(config):
            sys.exit(f"{out_dir} holds embeddings from a different config; move it to trash/ first")
        cfg.write_text(json.dumps({**config, "config_hash": config_hash(config), "git_commit": git_commit(),
                                   "device": device}, indent=2))
    done = set()
    shards = sorted(out_dir.glob("shard_*.npz"))
    for s in shards:
        done.update(np.load(s)["unit_ids"].tolist())
    todo = units[~units.unit_id.isin(done)]
    if a.limit:
        todo = todo.iloc[: a.limit]
    print(f"hear: {len(units)} units, {len(done)} done, {len(todo)} to go on {device}", flush=True)
    loader = DataLoader(Units(todo, paths), batch_size=a.batch, num_workers=a.workers, collate_fn=lambda b: b,
                        prefetch_factor=4 if a.workers else None)
    next_shard, buf_ids, buf_emb, errors, n, t0 = len(shards), [], [], [], 0, time.time()

    def flush():
        nonlocal next_shard, buf_ids, buf_emb
        if buf_ids and not a.limit:
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
                wins = torch.from_numpy(np.concatenate([w for _, w in ok])).to(device)
                owner = torch.from_numpy(np.repeat(np.arange(len(ok)), [len(w) for _, w in ok])).to(device)
                e = model(preprocess_audio(wins)).float()
                pooled = torch.zeros(len(ok), e.shape[1], device=device).index_add_(0, owner, e)
                pooled /= torch.bincount(owner, minlength=len(ok)).unsqueeze(1)
                buf_ids += [u for u, _ in ok]
                buf_emb.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
            n += len(batch)
            if len(buf_ids) >= a.shard or (a.limit and n >= len(todo)):
                if a.limit:
                    print(f"smoke: emb shape {np.vstack(buf_emb).shape}, finite {np.isfinite(np.vstack(buf_emb)).all()}")
                flush()
                r = n / (time.time() - t0)
                print(f"embedded {n}/{len(todo)}  {r:.1f} units/s  ETA {(len(todo) - n) / r / 60:.0f} min", flush=True)
    flush()
    if errors and not a.limit:
        with open(out_dir / "errors.jsonl", "a") as f:
            for u, e in errors:
                f.write(json.dumps({"unit_id": u, "error": e}) + "\n")
    print(f"done: {n} processed, {len(errors)} errors")


if __name__ == "__main__":
    main()
