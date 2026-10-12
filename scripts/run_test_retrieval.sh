#!/usr/bin/env bash
# Tier: cpu. Test-round retrieval (folds 1-4) with the per-task settings frozen on dev
# (artifacts/selection/retrieval_selection_dev.json). Nothing here looks at test labels to choose anything.
set -euo pipefail
cd "$(dirname "$0")/.."
SEL=${AUSCULTRAG_ARTIFACTS:-artifacts}/selection/retrieval_selection_dev.json
.venv/bin/python - "$SEL" <<'PY' | while read -r task center n m; do
import json, sys
for t, s in json.load(open(sys.argv[1])).items():
    print(t, s["center"], s["top_n"], s["m"])
PY
  echo "=== $task (center=$center N=$n m=$m)"
  .venv/bin/python scripts/run_retrieval.py --tasks "$task" --rounds 1,2,3,4 --scope corpus \
      --center "$center" --top-n "$n" --m "$m" --k 20
done
echo "=== test retrieval done"
