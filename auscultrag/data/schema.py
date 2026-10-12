"""The manifest schema (R1): one row per recording; the single source of truth for every stage."""
from __future__ import annotations

import pandas as pd

# Required columns and their pandas dtypes. Corpus-specific metadata lives in nullable extra
# columns (prefixed "meta_") so corpora stay separable without a wide sparse core schema.
CORE_COLUMNS: dict[str, str] = {
    "uid": "string",                 # globally unique: "<corpus>:<recording_id>"
    "corpus": "string",
    "family": "string",              # cough | respiratory | cardiac
    "patient_id": "string",          # "<corpus>:<resolved person id>" (identity-merged, see docs)
    "recording_id": "string",        # unique within corpus
    "device_id": "string",
    "dup_group_id": "string",        # recordings of the same physical event share this
    "filepath": "string",            # absolute path to the audio file
    "sample_rate": "Int64",
    "duration_sec": "Float64",
    "num_channels": "Int64",
    "content_sha1": "string",        # exact-duplicate detection across uids
    "label": "string",               # native label mapped into label_space_id; <NA> if none
    "label_space_id": "string",
    "has_closed_set_label": "boolean",
    "split": "string",               # round-1 view: knowledge_base | dev | test (see data/splits.py)
    "fold": "Int64",                 # patient-level fold 0..4 (0 = dev); <NA> for COUGHVID
}


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in CORE_COLUMNS.items()})


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Order core columns first, apply dtypes, keep meta_* columns after."""
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    extra = sorted(c for c in df.columns if c not in CORE_COLUMNS)
    bad = [c for c in extra if not c.startswith("meta_")]
    if bad:
        raise ValueError(f"non-core columns must be prefixed meta_: {bad}")
    out = df[list(CORE_COLUMNS) + extra].copy()
    for c, t in CORE_COLUMNS.items():
        out[c] = out[c].astype(t)
    # meta_* columns mix types across corpora; store them as nullable strings so parquet is stable.
    for c in extra:
        out[c] = out[c].map(_to_str).astype("string")
    return out


def _to_str(v):
    if _isna(v):
        return pd.NA
    if isinstance(v, float) and v.is_integer():
        return str(int(v))  # ints upcast to float by concat with NaN-bearing corpora
    return str(v)


def _isna(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False
