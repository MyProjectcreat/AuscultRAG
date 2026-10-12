"""Stage: make_text_queue. Tier: cpu. Text-only control conditions derived from an audio report queue.

Does the AUDIO contribute? For the same queries, two audio-free conditions are created:
  text_rag  : no audio; the model sees only the retrieved cases' expert annotations (same evidence block)
  text_none : no audio and no evidence; task description only (a prior / label-space baseline)
Prompts are the audio prompts with the audio sentence replaced; everything else is identical.

Usage: python scripts/make_text_queue.py --src <dir>/queue.jsonl --name report_test_v3_text
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402

AUDIO_SENT = re.compile(r"An audio clip is attached \(it may be short, about 1-15 seconds\)\. It is (.+?)\.\n")


def to_text(prompt: str, with_evidence: bool) -> str:
    m = AUDIO_SENT.search(prompt)
    if not m:
        raise ValueError("audio sentence not found in prompt")
    subject = m.group(1)
    repl = (f"No audio is available to you. The case is {subject}; you only have the reference information below.\n"
            if with_evidence else
            f"No audio is available to you and no other information about the case. The case is {subject}.\n")
    p = prompt[: m.start()] + repl + prompt[m.end():]
    p = p.replace("Write a short diagnostic report from the sound:", "Write a short diagnostic report:")
    p = p.replace("describing what you hear", "describing the most likely findings")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = ARTIFACT_ROOT / "llm" / a.name / "queue.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for line in a.src.open():
            it = json.loads(line)
            if it["condition"] == "rag":
                new = {**it, "condition": "text_rag", "no_audio": True, "prompt": to_text(it["prompt"], True)}
            else:
                new = {**it, "condition": "text_none", "no_audio": True, "prompt": to_text(it["prompt"], False)}
            new["item_id"] = f"{it['task']}|{new['condition']}|{it['uid']}"
            f.write(json.dumps(new) + "\n")
            n += 1
    print(f"wrote {out}: {n} items")


if __name__ == "__main__":
    main()
