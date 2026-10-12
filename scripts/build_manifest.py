"""Stage: build_manifest. Tier: cpu.

Runs every corpus builder, probes audio headers + content hashes in parallel, and writes the R1
manifest (one row per recording) with split = <NA>. Splits are assigned by scripts/make_splits.py.

Outputs:
  $AUSCULTRAG_ARTIFACTS/manifest/manifest_unsplit.parquet (+ .meta.json with build reports)
Usage: python scripts/build_manifest.py [--corpora coswara,tbscreen,icbhi,circor,coughvid] [--force]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, RAW_ROOT  # noqa: E402
from auscultrag.data.audio_probe import probe_many  # noqa: E402
from auscultrag.data.dedup import clean  # noqa: E402
from auscultrag.data.schema import coerce  # noqa: E402
from auscultrag.stage import cached, write_sidecar  # noqa: E402

ALL = ("coswara", "tbscreen", "icbhi", "circor", "coughvid", "sprsound", "kauh")
BUILDER_VERSION = 5  # bump when any builder's logic changes, to invalidate the cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default=",".join(ALL))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    corpora = [c for c in a.corpora.split(",") if c]
    out = ARTIFACT_ROOT / "manifest" / "manifest_unsplit.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"corpora": corpora, "raw_root": str(RAW_ROOT), "builder_version": BUILDER_VERSION}
    if cached(out, config) and not a.force:
        print(f"cached: {out}")
        return

    frames, reports = [], {}
    for name in corpora:
        mod = importlib.import_module(f"auscultrag.corpora.{name}")
        df, rep = mod.build(RAW_ROOT)
        reports[name] = rep
        print(f"[{name}] builder rows: {len(df)}; excluded: {dict(rep.excluded)}", flush=True)
        if len(df):
            probes = pd.DataFrame(probe_many(df["filepath"].tolist()))
            df = df.merge(probes, on="filepath", how="left", validate="one_to_one")
            bad = df["probe_error"].notna()
            for err in df.loc[bad, "probe_error"]:
                rep.drop("unreadable audio: " + err.split(":")[0])
            df = df.loc[~bad].drop(columns=["probe_error"])
            zero = df["duration_sec"].fillna(0) <= 0
            rep.drop("zero-length audio", int(zero.sum()))
            df = df.loc[~zero]
        frames.append(df)

    merged = clean(pd.concat(frames, ignore_index=True), reports)
    merged = merged.rename(columns={"peak_abs": "meta_peak_abs"})
    merged["fold"] = pd.NA  # assigned by make_splits.py
    for name in corpora:
        print(f"[{name}] manifest rows after content cleaning: {int((merged.corpus == name).sum())}", flush=True)
    reports = {k: v.to_dict() for k, v in reports.items()}

    manifest = coerce(merged)
    manifest = manifest.sort_values("uid").reset_index(drop=True)
    manifest.to_parquet(out, index=False)
    write_sidecar(out, stage="build_manifest", tier="cpu", config=config,
                  extra={"rows": len(manifest), "build_reports": reports})
    print(json.dumps(reports, indent=1, default=str))
    print(f"wrote {out} ({len(manifest)} rows)")


if __name__ == "__main__":
    main()
