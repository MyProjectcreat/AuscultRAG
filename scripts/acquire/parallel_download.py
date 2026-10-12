"""Tier: cpu. Resumable multi-connection HTTP Range download with md5 verification.

Some hosts (e.g. Zenodo) throttle per connection; fetching fixed-size chunks in parallel restores
throughput. Each chunk is written to <dest>.chunks/<index> and kept across restarts.

Usage: python scripts/acquire/parallel_download.py URL DEST --md5 HEX [--workers 8] [--chunk-mb 16]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def fetch_chunk(url: str, start: int, end: int, path: Path) -> None:
    want = end - start + 1
    if path.exists() and path.stat().st_size == want:
        return
    for attempt in range(10):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            if r.status_code == 206 and len(r.content) == want:
                tmp = path.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(path)
                return
        except requests.RequestException:
            pass
    raise RuntimeError(f"chunk {start}-{end} failed after retries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--md5", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=16)
    a = ap.parse_args()

    if a.dest.exists() and _md5(a.dest) == a.md5:
        print(f"{a.dest} already complete")
        return
    size = int(requests.head(a.url, allow_redirects=True, timeout=60).headers["Content-Length"])
    step = a.chunk_mb << 20
    ranges = [(i, s, min(s + step, size) - 1) for i, s in enumerate(range(0, size, step))]
    cdir = a.dest.with_name(a.dest.name + ".chunks")
    cdir.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(fetch_chunk, a.url, s, e, cdir / f"{i:06d}") for i, s, e in ranges]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 10 == 0 or done == len(ranges):
                print(f"chunks {done}/{len(ranges)}", flush=True)
    tmp = a.dest.with_suffix(a.dest.suffix + ".assembling")
    with tmp.open("wb") as out:
        for i, _, _ in ranges:
            with (cdir / f"{i:06d}").open("rb") as c:
                shutil.copyfileobj(c, out)
    got = _md5(tmp)
    if got != a.md5:
        sys.exit(f"md5 mismatch: got {got}, expected {a.md5}; chunks kept in {cdir}")
    tmp.replace(a.dest)
    print(f"md5 OK: {a.dest}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    main()
