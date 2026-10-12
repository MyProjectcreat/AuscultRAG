"""Classification metrics with patient-level bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def macro_f1(y_true: list, y_pred: list, labels: list) -> float:
    """Macro-F1 over classes present in y_true (a class absent from the truth contributes nothing);
    predictions outside `labels` (e.g. unparseable) count as wrong for every class."""
    present = [c for c in labels if c in set(y_true)]
    if not present:
        return float("nan")
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    f1s = []
    for c in present:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def balanced_accuracy(y_true: list, y_pred: list) -> float:
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    return float(np.mean([np.mean(yp[yt == c] == c) for c in np.unique(yt)]))


def summarise(df: pd.DataFrame, labels: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """df columns: patient_id, y_true, y_pred. CIs resample PATIENTS with replacement."""
    out = {
        "n_queries": int(len(df)), "n_patients": int(df.patient_id.nunique()),
        "accuracy": float(np.mean(df.y_true == df.y_pred)),
        "macro_f1": macro_f1(df.y_true.tolist(), df.y_pred.tolist(), labels),
        "balanced_accuracy": balanced_accuracy(df.y_true.tolist(), df.y_pred.tolist()),
        "invalid_rate": float(np.mean(~df.y_pred.isin(labels))),
    }
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("patient_id")}
    pids = np.array(list(groups))
    boots = {"accuracy": [], "macro_f1": []}
    for _ in range(n_boot):
        s = pd.concat([groups[p] for p in rng.choice(pids, size=len(pids), replace=True)])
        boots["accuracy"].append(np.mean(s.y_true == s.y_pred))
        boots["macro_f1"].append(macro_f1(s.y_true.tolist(), s.y_pred.tolist(), labels))
    for k, v in boots.items():
        out[f"{k}_ci95"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return out
