"""Stage: robustness_prepare. Tier: small_gpu (CLAP). Query-side perturbation study (R1-W3).

For a seeded subset of the frozen test queries (report_test_v3), each perturbation (auscultrag/robustness.py)
is applied to the whole query recording; the query's units (same boundaries as the clean run) are
re-encoded with CLAP under the exact clean signal contract, retrieved against the CLEAN knowledge base with
the task's frozen dev setting, and the corrupted audio is exported for the LLM. Writes one queue with
conditions "norag@<tag>" and "rag@<tag>" (report_score.py scores it unchanged and adds kNN on the new
evidence as "knn_rag@<tag>"). The clean tag reproduces the main run on the same subset.

Usage: python scripts/robustness_prepare.py --per-task 100 --name report_test_v3_robust
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auscultrag.config import ARTIFACT_ROOT, RAW_ROOT  # noqa: E402
from auscultrag.llm.prompts import render_v3  # noqa: E402
from auscultrag.llm.report_fields import FIELDS  # noqa: E402
from auscultrag.retrieval.engine import UnitIndex, retrieve_cases_multi  # noqa: E402
from auscultrag.retrieval.store import kb_mask, label_table, load_embeddings, load_tasks  # noqa: E402
from auscultrag.robustness import perturb  # noqa: E402
from embed_units import CLAP_SR, FAMILY_SR, MODEL_ID, REVISION  # noqa: E402
from llm_prepare import LLM_SR, MAX_AUDIO_SEC  # noqa: E402

PERTURBATIONS = [("clean", 0), ("babble", 20), ("babble", 10), ("babble", 5), ("babble", 0),
                 ("ambient", 20), ("ambient", 10), ("ambient", 5), ("ambient", 0),
                 ("white", 20), ("white", 10), ("white", 5), ("white", 0),
                 ("phone", 0), ("mp3", 8), ("clip", 0.3), ("clip", 0.1), ("trunc", 0.5), ("trunc", 0.25)]


def tag(kind, level) -> str:
    return kind if kind in ("clean", "phone") else f"{kind}{level}"


def _norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def llm_audio(x: np.ndarray, sr: int, start: float, out: Path, min_sec: float = 2.0) -> float:
    """Same export as llm_prepare.write_query_audio, from an in-memory (perturbed) signal."""
    a = int(start * sr)
    x = x[a: a + int(MAX_AUDIO_SEC * sr)]
    g = np.gcd(sr, LLM_SR)
    x = resample_poly(x, LLM_SR // g, sr // g)
    x = 0.9 * x / (np.max(np.abs(x)) or 1.0)
    short = int(min_sec * LLM_SR) - len(x)
    if short > 0:
        x = np.pad(x, (short // 2, short - short // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, x, LLM_SR, subtype="PCM_16")
    return len(x) / LLM_SR


def unit_signal(x: np.ndarray, sr: int, start: float, end: float, family: str) -> np.ndarray:
    """Same band-limit / resample as embed_units.load_unit, from an in-memory signal."""
    a, b, n_min = int(start * sr), int(end * sr), int(0.05 * sr)
    if b - a < n_min:                                   # truncated tail unit: keep at least 50 ms of real signal
        a = max(0, min(a, len(x) - n_min))
        b = a + n_min
    y = x[a:b]
    t = FAMILY_SR[family]
    g = np.gcd(sr, t)
    y = resample_poly(y, t // g, sr // g)
    return resample_poly(y, CLAP_SR // t, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ARTIFACT_ROOT / "llm" / "report_test_v3" / "queue.jsonl")
    ap.add_argument("--per-task", type=int, default=100)
    ap.add_argument("--name", default="report_test_v3_robust")
    ap.add_argument("--only", default="", help="comma list of tags to (re)build; default all")
    a = ap.parse_args()
    out_dir = ARTIFACT_ROOT / "llm" / a.name
    sel = json.loads((ARTIFACT_ROOT / "selection" / "retrieval_selection_dev.json").read_text())
    src = [json.loads(l) for l in a.src.open()]
    by_task = defaultdict(list)
    for it in src:
        if it["condition"] == "rag":
            by_task[it["task"]].append(it)
    rng = np.random.default_rng(20261010)
    queries = []
    for t, its in sorted(by_task.items()):
        its = sorted(its, key=lambda i: i["uid"])
        idx = rng.choice(len(its), size=min(a.per_task, len(its)), replace=False)
        queries += [its[i] for i in sorted(idx)]
    print(f"{len(queries)} queries", flush=True)

    tasks = load_tasks()
    labels = label_table()
    emb, units = load_embeddings("clap")
    units_full = pd.read_parquet(ARTIFACT_ROOT / "units" / "units.parquet")
    qunits = {u: g.sort_values("start_sec") for u, g in units_full[units_full.uid.isin({q["uid"] for q in queries})].groupby("uid")}
    gt = pd.read_parquet(ARTIFACT_ROOT / "reports" / "report_gt.parquet")
    gt_map = {(t, u): dict(zip(g.field, g.value)) for (t, u), g in gt.groupby(["task", "uid"], sort=False)}
    indexes = {}
    for t, r in sorted({(q["task"], q["round"]) for q in queries}):
        td, s = tasks[t], sel[t]
        mask = kb_mask(units, td["family"], r, td["corpus"], "corpus", False)
        kb = emb[mask]
        mu = kb.mean(axis=0, keepdims=True) if s["center"] else None
        indexes[(t, r)] = (UnitIndex(_norm(kb - mu) if s["center"] else kb, units[mask]), mu)
    print("indexes built", flush=True)

    from transformers import ClapModel, ClapProcessor
    cache = RAW_ROOT / "hf_models"
    model = ClapModel.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=cache).eval().to("cuda")
    proc = ClapProcessor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=cache)

    def encode(sigs: list[np.ndarray]) -> np.ndarray:
        outs = []
        with torch.inference_mode():
            for i in range(0, len(sigs), 64):
                inp = proc(audios=sigs[i: i + 64], sampling_rate=CLAP_SR, return_tensors="pt", padding="repeatpad")
                e = model.get_audio_features(**{k: v.to("cuda") for k, v in inp.items()})
                outs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.vstack(outs)

    only = set(a.only.split(",")) if a.only else None
    qpath = out_dir / "queue.jsonl"
    keep = []
    if qpath.exists() and only:                    # rebuild only the requested tags, keep the rest
        keep = [l for l in qpath.open() if json.loads(l)["condition"].split("@")[1] not in only]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with qpath.open("w") as f:
        for l in keep:
            f.write(l)
        for kind, level in PERTURBATIONS:
            tg = tag(kind, level)
            if only and tg not in only:
                continue
            sigs, owner, meta = [], [], []
            for q in queries:
                row = labels.loc[q["uid"]]
                x, sr = sf.read(row.filepath, dtype="float32", always_2d=True)
                x = perturb(x.mean(axis=1), sr, q["uid"], kind, level)
                dur = len(x) / sr
                us = qunits[q["uid"]]
                us = us[us.start_sec < dur] if (us.start_sec < dur).any() else us.iloc[:1]
                fam = tasks[q["task"]]["family"]
                for _, u in us.iterrows():
                    sigs.append(unit_signal(x, sr, u.start_sec, min(u.end_sec, dur), fam))
                    owner.append(len(meta))
                wav = out_dir / "audio" / tg / f"{q['uid'].replace(':', '__').replace('/', '_')}.wav"
                adur = llm_audio(x, sr, float(min(us.start_sec.min(), max(0.0, dur - 0.1))), wav)
                meta.append((q, wav, adur, row))
            E = encode(sigs)
            owner = np.array(owner)
            for i, (q, wav, adur, row) in enumerate(meta):
                t, s = q["task"], sel[q["task"]]
                index, mu = indexes[(t, q["round"])]
                qe = E[owner == i]
                if s["center"]:
                    qe = _norm(qe - mu)
                query = {"uid": q["uid"], "patient_id": q["patient_id"], "dup_group_id": row.dup_group_id}
                cases, diag = retrieve_cases_multi(index, qe, query, [s["top_n"]], [s["m"]], s["k"])[(s["top_n"], s["m"])]
                cf = [gt_map.get((t, c["uid"]), {}) for c in cases]
                rows.append({"tag": tg, "task": t, "uid": q["uid"], "n_units": int((owner == i).sum())})
                fields = FIELDS[t]
                for cond in ("norag", "rag"):
                    f.write(json.dumps({**q, "item_id": f"{t}|{cond}@{tg}|{q['uid']}", "condition": f"{cond}@{tg}",
                                        "perturbation": tg, "audio_path": str(wav), "audio_sec": round(adur, 3),
                                        "prompt": render_v3(t, fields, cond, cases, cf),
                                        "case_fields": cf, "case_scores": [c["score"] for c in cases],
                                        "case_uids": [c["uid"] for c in cases]}) + "\n")
            print(f"{tg}: {len(meta)} queries, {len(sigs)} units", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "units_per_query.csv", index=False)
    print(f"wrote {qpath}")


if __name__ == "__main__":
    main()
