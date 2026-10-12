"""Stage: make_extract_queue. Tier: cpu. Two-step protocol for open-weight audio-LLMs that do not follow the
REPORT-JSON format (AF3, Qwen2-Audio): step 1 is the model's own free-text reply (already in responses_<model>),
step 2 a NEUTRAL text-only extractor (Qwen2.5-7B-Instruct, run with llm_run_local.py) maps that reply to the
report fields. The extractor sees only the reply and the field definitions - never the audio, the evidence
block or the ground truth - and must answer "unknown"-free with the closest allowed value the reply supports.
Same rule for every source model.

Usage: python scripts/make_extract_queue.py --model af3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402
from auscultrag.llm.report_fields import FIELDS  # noqa: E402


def prompt(reply: str, fields: dict) -> str:
    spec = "\n".join(f'  "{f}": one of [{", ".join(vals)}]  -- {desc}' for f, (vals, desc) in fields.items())
    keys = ", ".join(f'"{f}": "..."' for f in fields)
    return ("You convert a free-text clinical audio report into structured fields. Use ONLY what the report below "
            "states or clearly implies; do not add medical knowledge of your own. For each field choose exactly one "
            "allowed value; if the report says nothing about a field, choose the value meaning absent/none/normal "
            "when one exists, otherwise the value most consistent with the report.\n\n"
            f"Report:\n\"\"\"\n{(reply or '').strip()[:3000]}\n\"\"\"\n\nFields:\n{spec}\n"
            f"Answer with ONE line only:\nREPORT: {{{keys}}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", type=Path, default=ARTIFACT_ROOT / "llm" / "report_test_v3" / "queue.jsonl")
    a = ap.parse_args()
    resp = {}
    for l in a.src.with_name(f"responses_{a.model}.jsonl").open():
        r = json.loads(l)
        if r.get("error") is None:
            resp[r["item_id"]] = r["text"]
    out = ARTIFACT_ROOT / "llm" / f"report_test_v3_extract_{a.model}" / "queue.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for l in a.src.open():
            it = json.loads(l)
            f.write(json.dumps({**it, "no_audio": True, "source_model": a.model,
                                "prompt": prompt(resp.get(it["item_id"], ""), FIELDS[it["task"]])}) + "\n")
            n += 1
    print(f"wrote {out}: {n} items")


if __name__ == "__main__":
    main()
