#!/usr/bin/env bash
# Tier: cpu. Encoder ablation (R1-W1, AI-W5): the same retrieval pipeline with a domain encoder in place of CLAP.
# Per encoder: dev grid (round 0) -> dev-only selection -> test retrieval (folds 1-4) with the frozen setting.
# Usage: scripts/run_encoder_ablation.sh operaCT hear
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for E in "$@"; do
  echo "=== $(date) $E: dev grid"
  for C in 0 1; do $PY scripts/run_retrieval.py --encoder $E --tasks all --rounds 0 --scope corpus --center $C --grid; done
  $PY scripts/select_retrieval_config.py --encoder $E
  SEL=${AUSCULTRAG_ARTIFACTS:-artifacts}/selection/retrieval_selection_dev_$E.json
  $PY - "$SEL" <<'PY' | while read -r task center n m; do
import json, sys
for t, s in json.load(open(sys.argv[1])).items():
    print(t, s["center"], s["top_n"], s["m"])
PY
    echo "=== $E $task (center=$center N=$n m=$m)"
    $PY scripts/run_retrieval.py --encoder $E --tasks "$task" --rounds 1,2,3,4 --scope corpus \
        --center "$center" --top-n "$n" --m "$m" --k 20
  done
  echo "=== $(date) $E test retrieval done"
done
