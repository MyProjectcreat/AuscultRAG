#!/usr/bin/env bash
# Tier: cpu. Corpus-identity analysis (reviewer objection 2, R8): rerun test-round retrieval with a
# FAMILY-wide knowledge base (all corpora of the acoustic family) using the dev-frozen per-task settings,
# and record how often retrieved cases come from the query's own corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
A=${AUSCULTRAG_ARTIFACTS:-artifacts}
until grep -q "test retrieval done" $A/logs/test_retrieval.log; do sleep 30; done
.venv/bin/python - $A/selection/retrieval_selection_dev.json <<'PY' | while read -r task center n m; do
import json, sys
for t, s in json.load(open(sys.argv[1])).items():
    if t != "circor_outcome":          # cardiac has a single corpus: family scope == corpus scope
        print(t, s["center"], s["top_n"], s["m"])
PY
  echo "=== $task"
  .venv/bin/python scripts/run_retrieval.py --tasks "$task" --rounds 1,2,3,4 --scope family \
      --center "$center" --top-n "$n" --m "$m" --k 20
done
echo "=== family scope done"
