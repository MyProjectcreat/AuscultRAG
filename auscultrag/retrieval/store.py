"""Loading embeddings + unit/recording tables, and building per-round knowledge bases."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from auscultrag.config import ARTIFACT_ROOT, REPO_ROOT

INDEX_COLS = ["unit_id", "uid", "patient_id", "dup_group_id", "corpus", "family", "fold", "split"]


def load_tasks() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "tasks.yaml").read_text())["tasks"]


def load_embeddings(encoder: str = "clap") -> tuple[np.ndarray, pd.DataFrame]:
    """All unit embeddings aligned with their unit rows (index-side columns only)."""
    d = ARTIFACT_ROOT / "embeddings" / encoder
    ids, embs = [], []
    for s in sorted(d.glob("shard_*.npz")):
        z = np.load(s, allow_pickle=False)
        ids.append(z["unit_ids"])
        embs.append(z["emb"])
    ids = np.concatenate(ids)
    emb = np.vstack(embs)
    units = pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet", columns=INDEX_COLS)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    units = units[units.unit_id.isin(pos.index)].reset_index(drop=True)
    emb = emb[pos[units.unit_id].to_numpy()]
    return emb, units


def label_table() -> pd.DataFrame:
    """Labels and metadata, kept apart from the index (joined only after case selection)."""
    return pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest.parquet").set_index("uid")


def kb_mask(units: pd.DataFrame, family: str, query_fold: int, query_corpus: str, scope: str,
            include_coughvid: bool) -> np.ndarray:
    """KB for one rotation round: same family; labelled units outside the query fold (all corpora when
    scope='family', the query corpus only when scope='corpus'); COUGHVID padding only if switched on.
    COUGHVID OOD-test units are never in any KB."""
    fam = units.family.to_numpy() == family
    labelled = units.fold.notna().to_numpy()
    out_fold = labelled & (units.fold.fillna(-1).to_numpy() != query_fold)
    if scope == "corpus":
        out_fold &= units.corpus.to_numpy() == query_corpus
    pad = (units.corpus.to_numpy() == "coughvid") & (units.split.to_numpy() == "knowledge_base") & include_coughvid
    return fam & (out_fold | pad)
