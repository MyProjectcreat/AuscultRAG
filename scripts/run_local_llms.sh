#!/usr/bin/env bash
# Tier: big_gpu. Open-weight audio-LLMs on the frozen test report queue (same prompts as the API models).
# Usage: scripts/run_local_llms.sh af3 qwen2-audio-7b      (resumable)
set -uo pipefail
cd "$(dirname "$0")/.."
GPY=${GPU_PYTHON:-${AUSCULTRAG_RAW:-raw_data}/envs/gpu/bin/python}
Q=${AUSCULTRAG_ARTIFACTS:-artifacts}/llm/report_test_v3/queue.jsonl
for M in "$@"; do
  for attempt in 1 2; do $GPY scripts/llm_run_local.py --queue $Q --model $M --batch ${BATCH:-8}; done
done
echo "=== local LLMs done"
