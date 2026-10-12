"""Stage: report_phase1. Tier: cpu. Summary statistics of the split manifest, written as Markdown + JSON
under artifacts_committed/ (committed to git).

Flags:
  * class with < 20 distinct patients in a (corpus, family)                       -> LOW_PATIENTS
  * task whose test pool (folds 1-4) has a class with < 20 patients               -> LOW
    (< 5 patients                                                                  -> SEVERE)
Usage: python scripts/report_phase1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, COMMITTED_ROOT, LABEL_SPACES, REPO_ROOT  # noqa: E402
from auscultrag.data.splits import DEV_FOLD, N_FOLDS  # noqa: E402
from auscultrag.stage import write_sidecar  # noqa: E402

LOW_PATIENTS, UNSTABLE, SEVERE = 20, 10, 5


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    m = pd.read_parquet(ARTIFACT_ROOT / "manifest" / "manifest.parquet")
    meta = json.loads((ARTIFACT_ROOT / "manifest" / "manifest_unsplit.parquet.meta.json").read_text())
    out, js = [], {}
    out.append("# Phase 1 data report\n")
    out.append(f"Manifest rows: **{len(m)}**; git commit of build: `{meta['git_commit']}`\n")

    # Corpus overview
    ov = m.groupby(["corpus", "family"]).agg(
        recordings=("uid", "size"), patients=("patient_id", "nunique"), dup_groups=("dup_group_id", "nunique"),
        sample_rates=("sample_rate", lambda s: ",".join(str(v) for v in sorted(s.unique()))),
        dur_median=("duration_sec", "median"), dur_p05=("duration_sec", lambda s: s.quantile(.05)),
        dur_p95=("duration_sec", lambda s: s.quantile(.95)), total_hours=("duration_sec", lambda s: s.sum() / 3600),
    ).round(3)
    out.append("## Corpus overview (recording-level)\n\n" + md_table(ov) + "\n")
    js["overview"] = ov.reset_index().to_dict("records")

    # Exclusions / notes from the builders
    out.append("## Builder exclusions and identity notes\n")
    for c, rep in meta["build_reports"].items():
        out.append(f"**{c}**: excluded {json.dumps(rep['excluded'])}  ")
        for n in rep["notes"]:
            out.append(f"- {n}")
        out.append("")

    # Dup groups
    dg = m.groupby(["corpus", "dup_group_id"]).size().rename("files").reset_index()
    dgs = dg.groupby("corpus")["files"].agg(groups="size", singleton=lambda s: int((s == 1).sum()),
                                           mean="mean", max="max").round(2)
    size_dist = dg.groupby("corpus")["files"].value_counts().unstack(fill_value=0)
    out.append("## Near-duplicate groups\n\n" + md_table(dgs) + "\n\nGroup-size distribution (files per group):\n\n"
               + md_table(size_dist) + "\n")
    tb = m[m.corpus == "tbscreen"]
    if len(tb):
        dev_per = tb.groupby("dup_group_id")["device_id"].nunique().value_counts().sort_index()
        out.append("TBScreen: distinct microphones per cough event:\n\n" + md_table(dev_per.rename("cough_events").to_frame()) + "\n")
        js["tbscreen_devices_per_event"] = dev_per.to_dict()

    # Label distributions
    out.append("## Label distribution per (corpus, family)\n")
    js["labels"] = {}
    lab = m[m.has_closed_set_label.fillna(False)]
    for (c, f), sub in lab.groupby(["corpus", "family"]):
        space = LABEL_SPACES[sub.label_space_id.iloc[0]]
        t = pd.DataFrame({"recordings": sub.label.value_counts(), "patients": sub.groupby("label").patient_id.nunique()})
        t = t.reindex(space).fillna(0).astype(int)
        t["flag"] = t.patients.map(lambda p: "LOW_PATIENTS" if p < LOW_PATIENTS else "")
        out.append(f"### {c} / {f} (label space `{sub.label_space_id.iloc[0]}`)\n\n" + md_table(t) + "\n")
        js["labels"][f"{c}/{f}"] = t.to_dict("index")

    # Folds per task (rotating-test design, auscultrag/data/splits.py)
    tasks = yaml.safe_load((REPO_ROOT / "configs" / "tasks.yaml").read_text())["tasks"]
    out.append("## Folds per task (fold 0 = dev; folds 1-4 rotate as test, KB = all other folds)\n")
    js["tasks"] = {}
    for tname, t in tasks.items():
        sub = m[(m.corpus == t["corpus"]) & (m.family == t["family"])]
        sub = sub[sub[t["label_field"]].notna()]
        lf = t["label_field"]
        space = list(LABEL_SPACES[t["label_space"]])
        pt = sub.groupby(["fold", lf]).patient_id.nunique().unstack(fill_value=0).reindex(columns=space, fill_value=0)
        rc = sub.groupby(["fold", lf]).size().unstack(fill_value=0).reindex(columns=space, fill_value=0)
        pt.index = [f"fold {int(i)}" + (" (dev)" if int(i) == DEV_FOLD else "") for i in pt.index]
        rc.index = pt.index
        test_pool = sub[sub.fold != DEV_FOLD].groupby(lf).patient_id.nunique().reindex(space, fill_value=0)
        dev = sub[sub.fold == DEV_FOLD].groupby(lf).patient_id.nunique().reindex(space, fill_value=0)
        pt.loc["TEST POOL (folds 1-4)"] = test_pool
        flags = []
        for lab, n in test_pool.items():
            if n < SEVERE:
                flags.append(f"SEVERE: test pool has {n} patients of {lab}")
            elif n < LOW_PATIENTS:
                flags.append(f"LOW: test pool has {n} patients of {lab}")
        for lab, n in dev.items():
            if n < SEVERE:
                flags.append(f"dev fold has {n} patients of {lab} (tuning signal weak)")
        out.append(f"### {tname} (`{t['label_space']}`, label field `{lf}`): {sub.patient_id.nunique()} patients, {len(sub)} recordings\n")
        out.append("Patients per label:\n\n" + md_table(pt) + "\n\nRecordings per label:\n\n" + md_table(rc) + "\n")
        if flags:
            out.append("**Flags:** " + "; ".join(flags) + "\n")
        js["tasks"][tname] = {"patients_per_fold_label": pt.to_dict("index"), "flags": flags}
    cv = m[m.corpus == "coughvid"]
    if len(cv):
        out.append(f"COUGHVID: {int((cv.split == 'test').sum())} expert-annotated recordings in the test-only OOD pool; "
                   f"{int((cv.split == 'knowledge_base').sum())} unlabelled recordings available as switchable KB padding.\n")

    # Disjointness proof, per rotation round
    out.append("## Patient-disjointness proof (every rotation round, every corpus)\n")
    proof = {}
    lab = m[m.fold.notna()]
    for c, sub in lab.groupby("corpus"):
        for q in range(N_FOLDS):
            qs, kb = sub[sub.fold == q], sub[sub.fold != q]
            name = "tuning" if q == DEV_FOLD else f"test round {q}"
            proof[f"{c}: {name}, patients in query ∩ KB"] = len(set(qs.patient_id) & set(kb.patient_id))
    cv = m[m.corpus == "coughvid"]
    proof["coughvid: OOD test ∩ KB patients"] = len(set(cv[cv.split == "test"].patient_id) & set(cv[cv.split != "test"].patient_id))
    out.append(md_table(pd.Series(proof, name="intersection").to_frame()) + "\n")
    js["pid_intersections"] = proof

    COMMITTED_ROOT.mkdir(parents=True, exist_ok=True)
    md = COMMITTED_ROOT / "phase1_report.md"
    md.write_text("\n".join(out))
    (COMMITTED_ROOT / "phase1_report.json").write_text(json.dumps(js, indent=1, default=str))
    write_sidecar(md, stage="report_phase1", tier="cpu", config={"thresholds": [LOW_PATIENTS, UNSTABLE, SEVERE]})
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
