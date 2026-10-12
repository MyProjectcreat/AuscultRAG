"""Stage: llm_run. Tier: api (network only). Sends a prepared queue to an audio LLM:
OpenAI audio models (gpt-audio*) via Chat Completions, Google Gemini (gemini-*) via google-genai.

Append-only results keyed by item_id (R11): on start, finished items are skipped; a kill loses at most
the in-flight requests. Retries transient errors with backoff; permanent failures are recorded as
errors (never silently dropped). Concurrency via a small thread pool.

Usage: python scripts/llm_run.py --queue <dir>/queue.jsonl --model gpt-audio-2025-08-28 [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auscultrag.config  # noqa: E402,F401  (loads API keys from the project's .env)

_local = threading.local()


def call_gemini(model: str, item: dict, max_tries: int = 6) -> dict:
    from google import genai
    from google.genai import types
    if not hasattr(_local, "gemini"):
        import os
        _local.gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    audio = None if item.get("no_audio") else Path(item["audio_path"]).read_bytes()
    delay, err = 5.0, None
    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            r = _local.gemini.models.generate_content(
                model=model, contents=([types.Part.from_bytes(data=audio, mime_type="audio/wav")] if audio else []) + [item["prompt"]])
            usage = r.usage_metadata.model_dump() if r.usage_metadata else None
            finish = r.candidates[0].finish_reason.name if r.candidates and r.candidates[0].finish_reason else None
            return {"item_id": item["item_id"], "model": model, "response_model": model, "text": r.text,
                    "finish_reason": finish, "usage": usage, "latency_sec": round(time.time() - t0, 2),
                    "attempts": attempt, "error": None}
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            code = getattr(exc, "code", None)
            if isinstance(code, int) and code < 500 and code != 429:
                break
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return {"item_id": item["item_id"], "model": model, "text": None, "error": err, "attempts": attempt}


def call(client: OpenAI, model: str, item: dict, max_tries: int = 6) -> dict:
    content = [{"type": "text", "text": item["prompt"]}]
    if not item.get("no_audio"):
        audio_b64 = base64.b64encode(Path(item["audio_path"]).read_bytes()).decode()
        content.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}})
    messages = [{"role": "user", "content": content}]
    extra = {} if item.get("no_audio") else {"modalities": ["text"]}
    delay = 5.0
    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=model, messages=messages, **extra)
            return {"item_id": item["item_id"], "model": model, "response_model": r.model,
                    "text": r.choices[0].message.content, "finish_reason": r.choices[0].finish_reason,
                    "usage": r.usage.model_dump() if r.usage else None, "latency_sec": round(time.time() - t0, 2),
                    "attempts": attempt, "error": None}
        except Exception as exc:  # transient errors are retried; the last one is recorded
            err = f"{type(exc).__name__}: {exc}"
            status = getattr(exc, "status_code", None)
            if status is not None and status < 500 and status != 429:
                break
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return {"item_id": item["item_id"], "model": model, "text": None, "error": err, "attempts": attempt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--model", default="gpt-audio-2025-08-28")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="send at most N new items (smoke test)")
    a = ap.parse_args()
    out = a.queue.with_name(f"responses_{a.model}.jsonl")
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            if r.get("error") is None:
                done.add(r["item_id"])
    items = [json.loads(l) for l in a.queue.open()]
    todo = [it for it in items if it["item_id"] not in done][: a.limit]
    print(f"{len(items)} items, {len(done)} done, sending {len(todo)} to {a.model}", flush=True)
    client = None if a.model.startswith("gemini") else OpenAI()
    lock = threading.Lock()
    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool, out.open("a") as f:
        if a.model.startswith("gemini"):
            futs = [pool.submit(call_gemini, a.model, it) for it in todo]
        else:
            futs = [pool.submit(call, client, a.model, it) for it in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                f.write(json.dumps(r) + "\n")
                f.flush()
            n_ok += r["error"] is None
            n_err += r["error"] is not None
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"{i}/{len(todo)} ok={n_ok} err={n_err} {rate:.2f}/s ETA {(len(todo) - i) / rate / 60:.1f} min", flush=True)
    if n_err:
        print(f"{n_err} items failed; re-run to retry them", file=sys.stderr)


if __name__ == "__main__":
    main()
