"""Stage: make_splits. Tier: cpu.

Reads manifest_unsplit.parquet, assigns patient-level folds (fold 0 = dev, folds 1-4 rotate as test;
see auscultrag/data/splits.py) plus the round-1 `split` view, writes the final
manifest, the patient->split table, and committed copies under artifacts_committed/ (git is the
durable store for these small, irreplaceable files).

Usage: python scripts/make_splits.py [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, COMMITTED_ROOT, SEED  # noqa: E402
from auscultrag.data.schema import coerce  # noqa: E402
from auscultrag.data.splits import DEV_FOLD, N_FOLDS, split_manifest  # noqa: E402
from auscultrag.stage import cached, config_hash, write_sidecar  # noqa: E402

POLICY = {"coughvid": "coughvid_expert_test"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src = ARTIFACT_ROOT / "manifest" / "manifest_unsplit.parquet"
    src_meta = src.with_name(src.name + ".meta.json").read_text()
    out = ARTIFACT_ROOT / "manifest" / "manifest.parquet"
    config = {"n_folds": N_FOLDS, "dev_fold": DEV_FOLD, "seed": SEED, "policy": POLICY, "source_meta_hash": config_hash({"m": src_meta})}
    if cached(out, config) and not a.force:
        print(f"cached: {out}")
        return

    manifest = pd.read_parquet(src)
    split, patients = split_manifest(manifest, SEED, POLICY)
    split = coerce(split)
    split.to_parquet(out, index=False)
    write_sidecar(out, stage="make_splits", tier="cpu", config=config)
    ptab = ARTIFACT_ROOT / "manifest" / "patient_splits.csv"
    patients.to_csv(ptab, index=False)
    write_sidecar(ptab, stage="make_splits", tier="cpu", config=config)

    COMMITTED_ROOT.mkdir(parents=True, exist_ok=True)
    split.to_csv(COMMITTED_ROOT / "manifest.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    for f in (ptab, ptab.with_name(ptab.name + ".meta.json"), out.with_name(out.name + ".meta.json"),
              src.with_name(src.name + ".meta.json")):
        shutil.copy2(f, COMMITTED_ROOT / f.name)
    print(f"wrote {out} and committed copies under {COMMITTED_ROOT}")


if __name__ == "__main__":
    main()
