"""Header-level audio probing (sample rate, duration, channels) plus content hashing, parallelised."""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf


def probe_one(path: str) -> dict:
    out = {"filepath": path, "sample_rate": None, "duration_sec": None, "num_channels": None,
           "content_sha1": None, "peak_abs": None, "probe_error": None}
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out["content_sha1"] = h.hexdigest()
        info = sf.info(path)
        out["sample_rate"] = int(info.samplerate)
        out["num_channels"] = int(info.channels)
        out["duration_sec"] = float(info.frames) / float(info.samplerate) if info.samplerate else None
        # Peak amplitude, read in blocks so long recordings do not load fully into memory.
        peak = 0.0
        for block in sf.blocks(path, blocksize=1 << 18, dtype="float32", always_2d=True):
            if block.size:
                peak = max(peak, float(abs(block).max()))
        out["peak_abs"] = peak
    except Exception as exc:  # recorded per-file; callers decide whether to drop the row
        out["probe_error"] = f"{type(exc).__name__}: {exc}"
    return out


def probe_many(paths: list[str], workers: int | None = None) -> list[dict]:
    workers = workers or len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(probe_one, paths, chunksize=64))


def exists_readable(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and os.access(p, os.R_OK)
