"""Tier: cpu. Transcode COUGHVID webm/ogg/wav to 16-bit PCM wav at the decoder's native rate.

Lossless with respect to the decoded signal (no resampling, no channel mixing), so later stages read
every corpus through soundfile. Resumable: existing outputs are skipped; writes are atomic.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SRC_DIR = "coughvid_20211012"
OUT_DIR = "wav_pcm16"


def transcode(args: tuple[str, str]) -> str | None:
    src, dst = args
    if os.path.exists(dst):
        return None
    tmp = dst + ".part.wav"
    proc = subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", src, "-c:a", "pcm_s16le", tmp],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.replace(tmp, dst + ".failed")
        return f"{src}: {proc.stderr.strip()[:300]}"
    os.replace(tmp, dst)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    a = ap.parse_args()
    src_root = a.raw_root / "coughvid" / SRC_DIR
    out_root = a.raw_root / "coughvid" / OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(out_root / (p.stem + ".wav"))) for p in sorted(src_root.iterdir())
            if p.suffix in (".webm", ".ogg", ".wav")]
    stems = [Path(s).stem for s, _ in jobs]
    if len(stems) != len(set(stems)):
        raise SystemExit("COUGHVID: the same uuid has more than one audio file; refusing to guess which to keep")
    errors = []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, err in enumerate(pool.map(transcode, jobs, chunksize=32), 1):
            if err:
                errors.append(err)
            if i % 2000 == 0:
                print(f"transcoded {i}/{len(jobs)} (errors {len(errors)})", flush=True)
    (out_root / "_transcode_errors.txt").write_text("\n".join(errors))
    print(f"COUGHVID transcode done: {len(jobs)} files, {len(errors)} errors")


if __name__ == "__main__":
    main()
