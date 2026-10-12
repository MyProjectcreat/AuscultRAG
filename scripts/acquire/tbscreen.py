"""Tier: cpu. Selectively re-acquire TBScreen raw audio + metadata from the 395 GB archive.

Source: https://dataset.cs.washington.edu/shwetak/TBscreen_Dataset.zip (linked from Zenodo
record 10431329, CC-BY-4.0). Only Audio_files/*.wav and the CSV/README metadata are fetched via
HTTP Range requests (~2.5 GB); scalogram / mel / VGGish feature arrays are skipped. Each member
is CRC-checked on extraction. Resumable: members already on disk with the right size are skipped.

Usage: python scripts/acquire/tbscreen.py [--raw-root DIR] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from auscultrag.corpora.remote_zip import RemoteZip, ZipMember  # noqa: E402

URL = "https://dataset.cs.washington.edu/shwetak/TBscreen_Dataset.zip"
EXPECTED_ARCHIVE_BYTES = 424_584_306_004
METADATA = {
    "TBscreen_Dataset/README.md",
    "TBscreen_Dataset/Metadata.csv",
    "TBscreen_Dataset/Dataset_key.xlsx",
    "TBscreen_Dataset/Passive_coughs/Passive_coughs.csv",
    "TBscreen_Dataset/Forced_coughs/Forced_coughs.csv",
}


def wanted(name: str) -> bool:
    return name in METADATA or ("/Audio_files/" in name and name.endswith(".wav"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=os.environ.get("AUSCULTRAG_RAW", str(Path(__file__).resolve().parents[2] / "raw_data")))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.raw_root) / "tbscreen"
    root.mkdir(parents=True, exist_ok=True)

    rz = RemoteZip(URL)
    if rz.size != EXPECTED_ARCHIVE_BYTES:
        sys.exit(f"Archive size changed upstream: {rz.size} != {EXPECTED_ARCHIVE_BYTES}; refusing to proceed.")
    cd_path = root / "central_directory.jsonl"
    if cd_path.exists():
        members = [ZipMember(**json.loads(line)) for line in cd_path.open()]
    else:
        members = rz.members()
        with cd_path.open("w") as f:
            for m in members:
                f.write(json.dumps(m.__dict__) + "\n")

    todo = []
    for m in members:
        if not wanted(m.name):
            continue
        dest = root / m.name
        if dest.exists() and dest.stat().st_size == m.file_size:
            continue
        todo.append((m, dest))
    total = sum(1 for m in members if wanted(m.name))
    print(f"{total} wanted members, {total - len(todo)} already present, {len(todo)} to fetch "
          f"({sum(m.compressed_size for m, _ in todo) / 1e9:.2f} GB)", flush=True)

    local = threading.local()

    def fetch(m: ZipMember, dest: Path) -> None:
        # requests.Session is not thread-safe; give each worker its own connection.
        if not hasattr(local, "rz"):
            local.rz = RemoteZip(URL)
        local.rz.extract(m, dest)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, m, dest): m for m, dest in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as exc:  # report and continue; a re-run picks up failures
                failed += 1
                print(f"FAILED {futs[fut].name}: {exc}", flush=True)
            if (done + failed) % 1000 == 0:
                print(f"progress {done + failed}/{len(todo)} (failed {failed})", flush=True)
    print(f"done: fetched {done}, failed {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
