"""Query-side perturbations for the robustness study (R1-W3). The knowledge base stays clean (deployment
setting: a curated case library, noisy point-of-care queries). Every perturbation is applied to the WHOLE
query recording at its native sample rate, before unit cutting / encoding / LLM audio export, and is seeded
by (uid, kind, level) so all models and conditions hear exactly the same corrupted audio.

kinds (levels):
  babble  (SNR dB)  : sum of 4 random LibriSpeech dev-clean utterances (crowded clinic / talking)
  ambient (SNR dB)  : ESC-50 indoor/clinic-like sounds (footsteps, door, keyboard, mouse, clock, appliances)
  white   (SNR dB)  : white Gaussian noise
  phone   (0)       : telephone channel, 300-3400 Hz band-pass
  mp3     (kbps)    : MP3 round trip at low bit rate (libsndfile)
  clip    (fraction): hard clipping at the given fraction of the peak
  trunc   (fraction): keep only the first fraction of the recording
"""
from __future__ import annotations

import io
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from auscultrag.config import RAW_ROOT

NOISE = RAW_ROOT / "noise"
AMBIENT_CLASSES = ["footsteps", "door_wood_knock", "door_wood_creaks", "keyboard_typing", "mouse_click",
                   "clock_tick", "vacuum_cleaner", "washing_machine", "water_drops", "toilet_flush"]
NOISE_SR = 16000


def _rng(uid: str, kind: str, level) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{uid}|{kind}|{level}".encode()))


@lru_cache(maxsize=1)
def _speech_pool() -> list[np.ndarray]:
    files = sorted((NOISE / "librispeech" / "LibriSpeech" / "dev-clean").rglob("*.flac"))
    rng = np.random.default_rng(0)
    return [sf.read(str(f), dtype="float32")[0] for f in rng.choice(files, size=300, replace=False)]


@lru_cache(maxsize=1)
def _ambient_pool() -> list[np.ndarray]:
    root = next((NOISE / "esc50").glob("*"))
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    out = []
    for f in meta[meta.category.isin(AMBIENT_CLASSES)].filename:
        x, sr = sf.read(str(root / "audio" / f), dtype="float32")
        g = np.gcd(NOISE_SR, sr)
        out.append(resample_poly(x, NOISE_SR // g, sr // g).astype(np.float32))
    return out


def _fill(pool: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(0, dtype=np.float32)
    while len(out) < n:
        out = np.concatenate([out, pool[rng.integers(len(pool))]])
    s = rng.integers(0, len(out) - n + 1)
    return out[s: s + n]


def _to_sr(x: np.ndarray, sr: int) -> np.ndarray:
    g = np.gcd(NOISE_SR, sr)
    return resample_poly(x, sr // g, NOISE_SR // g)


def _mix(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps, pn = np.mean(x ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    return (x + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def perturb(x: np.ndarray, sr: int, uid: str, kind: str, level) -> np.ndarray:
    rng = _rng(uid, kind, level)
    n = len(x)
    if kind == "clean":
        return x
    if kind == "white":
        return _mix(x, rng.standard_normal(n).astype(np.float32), level)
    if kind == "babble":
        n16 = int(np.ceil(n * NOISE_SR / sr)) + 1
        b = sum(_fill(_speech_pool(), n16, rng) for _ in range(4))
        return _mix(x, _to_sr(b, sr)[:n], level)
    if kind == "ambient":
        n16 = int(np.ceil(n * NOISE_SR / sr)) + 1
        return _mix(x, _to_sr(_fill(_ambient_pool(), n16, rng), sr)[:n], level)
    if kind == "phone":
        hi = min(3400, 0.45 * sr)
        return sosfiltfilt(butter(4, [300, hi], btype="band", fs=sr, output="sos"), x).astype(np.float32)
    if kind == "mp3":
        x8 = resample_poly(x, 8000, sr) if sr != 8000 else x      # libsndfile MP3 needs a standard rate
        buf = io.BytesIO()
        peak = np.max(np.abs(x8)) or 1.0
        # compression_level 0.9 (1.0 is rejected by libsndfile) = very low bit rate; level is a nominal tag for naming
        sf.write(buf, x8 / peak * 0.9, 8000, format="MP3", compression_level=0.9)
        buf.seek(0)
        y = sf.read(buf, dtype="float32")[0] * peak / 0.9
        y = resample_poly(y, sr, 8000) if sr != 8000 else y
        return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)
    if kind == "clip":
        t = level * (np.max(np.abs(x)) or 1.0)
        return np.clip(x, -t, t).astype(np.float32)
    if kind == "trunc":
        return x[: max(1, int(n * level))]
    raise ValueError(kind)
