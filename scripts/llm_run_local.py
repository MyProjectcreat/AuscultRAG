"""Stage: llm_run_local. Tier: big_gpu. Open-weight audio-LLMs on the SAME queue as the API models
(identical prompts, audio and evidence), so report_score.py scores them unchanged.

Models (HF ids; greedy decoding, max_new_tokens 512):
  qwen2-audio-7b  Qwen/Qwen2-Audio-7B-Instruct   (model used in the original paper)
  af3             nvidia/audio-flamingo-3-hf     (Audio Flamingo 3)
  medgemma-4b     google/medgemma-4b-it          (medical LLM; text-only queues, e.g. report_test_v3_text)
  medgemma-27b    google/medgemma-27b-text-it    (medical LLM, text-only)
  qwen2.5-7b      Qwen/Qwen2.5-7B-Instruct       (neutral text extractor for the two-step protocol)
Runs in the separate GPU env (envs/gpu, transformers 5.x):
  $GPU_PYTHON scripts/llm_run_local.py --queue <dir>/queue.jsonl --model af3
Output: <dir>/responses_<model>.jsonl (resumable; same record format as llm_run.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import RAW_ROOT  # noqa: E402

MODELS = {"qwen2-audio-7b": ("Qwen/Qwen2-Audio-7B-Instruct", "Qwen2AudioForConditionalGeneration"),
          "af3": ("nvidia/audio-flamingo-3-hf", "AudioFlamingo3ForConditionalGeneration"),
          "medgemma-4b": ("google/medgemma-4b-it", "Gemma3ForConditionalGeneration"),
          "medgemma-27b": ("google/medgemma-27b-text-it", "AutoModelForCausalLM"),
          "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "AutoModelForCausalLM")}
TEXT_ONLY = {"medgemma-4b", "medgemma-27b", "qwen2.5-7b"}
MAX_NEW = 512


def load(name: str):
    import transformers
    repo, cls = MODELS[name]
    kw = {"cache_dir": RAW_ROOT / "hf_models"}
    if name in TEXT_ONLY:          # tokenizer only (no image / audio processor needed)
        proc = transformers.AutoTokenizer.from_pretrained(repo, **kw)
        proc.padding_side = "left"
    else:
        proc = transformers.AutoProcessor.from_pretrained(repo, **kw)
        proc.tokenizer.padding_side = "left"
    model = getattr(transformers, cls).from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
    return proc, model


def build(proc, items: list[dict], text_only: bool = False):
    if text_only:
        if not all(it.get("no_audio") for it in items):
            raise ValueError("text-only model given an audio item")
        text = [proc.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) for it in items]
        return proc(text, return_tensors="pt", padding=True, add_special_tokens=False)
    convs = []
    for it in items:
        content = [{"type": "text", "text": it["prompt"]}]
        if not it.get("no_audio"):
            content = [{"type": "audio", "path": it["audio_path"]}] + content
        convs.append([{"role": "user", "content": content}])
    return proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                    return_tensors="pt", processor_kwargs={"text_kwargs": {"padding": True}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(l) for l in a.queue.open()]
    out = a.queue.with_name(f"responses_{a.model}.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["item_id"] for l in out.open() if json.loads(l).get("error") is None}
    todo = [it for it in items if it["item_id"] not in done][: a.limit or None]
    print(f"{len(items)} items, {len(done)} done, running {len(todo)} on {a.model}", flush=True)
    proc, model = load(a.model)
    t0 = time.time()
    with out.open("a") as f, torch.inference_mode():
        for i in range(0, len(todo), a.batch):
            chunk = todo[i: i + a.batch]
            tb = time.time()
            try:
                inp = build(proc, chunk, a.model in TEXT_ONLY).to(model.device)
                inp = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
                gen = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
                texts = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                recs = [{"item_id": it["item_id"], "model": a.model, "response_model": MODELS[a.model][0], "text": t,
                         "latency_sec": round((time.time() - tb) / len(chunk), 2), "error": None}
                        for it, t in zip(chunk, texts)]
            except Exception as exc:
                recs = [{"item_id": it["item_id"], "model": a.model, "text": None,
                         "error": f"{type(exc).__name__}: {exc}"} for it in chunk]
                torch.cuda.empty_cache()
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n = i + len(chunk)
            r = n / (time.time() - t0)
            print(f"{n}/{len(todo)}  {r:.2f} items/s  ETA {(len(todo) - n) / r / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
