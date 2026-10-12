"""The R14 verifier must fail loudly on each kind of violation (a passing run proves nothing otherwise)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from auscultrag.data.schema import coerce  # noqa: E402

SCRIPT = REPO / "scripts" / "verify_invariants.py"


def make_valid(tmp: Path) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    folds = [2, 0, 1, 3, 4, 0, 1, 2, 3, 4]
    view = {0: "dev", 1: "test"}
    for i, fold in enumerate(folds):
        split = view.get(fold, "knowledge_base")
        f = tmp / f"a{i}.wav"
        sf.write(f, rng.standard_normal(800).astype("float32") * 0.1, 8000)
        rows.append({
            "uid": f"circor:r{i}", "corpus": "circor", "family": "cardiac", "patient_id": f"circor:p{i}",
            "recording_id": f"r{i}", "device_id": "unknown", "dup_group_id": f"circor:r{i}", "filepath": str(f),
            "sample_rate": 8000, "duration_sec": 0.1, "num_channels": 1, "content_sha1": f"sha{i}",
            "label": "Normal", "label_space_id": "circor_outcome2", "has_closed_set_label": True, "split": split, "fold": fold,
        })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, tmp: Path) -> int:
    path = tmp / "m.parquet"
    coerce(df).to_parquet(path, index=False)
    return subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(path), "--allow-missing-tasks"], capture_output=True).returncode


def test_valid_manifest_passes(tmp_path):
    assert run(make_valid(tmp_path), tmp_path) == 0


@pytest.mark.parametrize("mutation", [
    "patient_overlap", "recording_overlap", "dup_group_split", "label_outside_space",
    "missing_file", "shared_file", "duplicate_content", "missing_split", "dup_group_two_folds", "split_view_mismatch",
])
def test_each_violation_fails(tmp_path, mutation):
    df = make_valid(tmp_path)
    if mutation == "patient_overlap":
        df.loc[1, "patient_id"] = df.loc[0, "patient_id"]          # KB patient also in dev
    elif mutation == "recording_overlap":
        df.loc[1, "recording_id"] = df.loc[0, "recording_id"]
    elif mutation == "dup_group_split":
        df.loc[1, "dup_group_id"] = df.loc[0, "dup_group_id"]
    elif mutation == "label_outside_space":
        df.loc[0, "label"] = "Murmur"
    elif mutation == "missing_file":
        df.loc[0, "filepath"] = str(tmp_path / "nope.wav")
    elif mutation == "shared_file":
        df.loc[1, "filepath"] = df.loc[0, "filepath"]
    elif mutation == "duplicate_content":
        df.loc[1, "content_sha1"] = df.loc[0, "content_sha1"]
    elif mutation == "missing_split":
        df.loc[0, "split"] = pd.NA
    elif mutation == "dup_group_two_folds":
        df.loc[3, "dup_group_id"] = df.loc[4, "dup_group_id"]      # both knowledge_base in round 1
    elif mutation == "split_view_mismatch":
        df.loc[3, "split"] = "test"
    assert run(df, tmp_path) != 0
