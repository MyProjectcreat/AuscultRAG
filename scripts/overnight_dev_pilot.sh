#!/usr/bin/env bash
# Unattended dev pipeline (2026-10-09 night): wait for the dev retrieval grid -> select retrieval config
# on dev -> prepare LLM queue -> run gpt-audio pilot (dev only, ~1,600 calls, approved budget) -> score
# -> copy small results into artifacts_committed/. Every stage is resumable; re-running this script continues.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=${AUSCULTRAG_ARTIFACTS:-artifacts}
LOG=$A/logs/overnight_dev_pilot.log
MODEL=gpt-audio-2025-08-28
OUT=artifacts_committed/phase2_dev
exec >> "$LOG" 2>&1
echo "=== $(date) start"

until [ "$(grep -c '^wrote' $A/logs/retrieval_dev_grid.log)" -ge 2 ]; do
  if grep -q Traceback $A/logs/retrieval_dev_grid.log; then echo "grid failed"; exit 1; fi
  sleep 30
done
echo "=== $(date) grid done"

$PY scripts/select_retrieval_config.py
$PY scripts/llm_prepare.py --round 0 --per-task 100 --selection $A/selection/retrieval_selection_dev.json
Q=$A/llm/round0_n100_dev-selected/queue.jsonl
echo "=== $(date) queue: $(wc -l < $Q) items"

for attempt in 1 2 3; do               # re-runs only retry failed / missing items
  $PY scripts/llm_run.py --queue $Q --model $MODEL --workers 4
done
$PY scripts/llm_score.py --queue $Q --model $MODEL

mkdir -p $OUT
cp $A/retrieval/clap/summary_dev_grid_scope-corpus_cv-0_c*.csv $OUT/
cp $A/selection/retrieval_selection_dev.json $OUT/
cp $(dirname $Q)/scores_$MODEL.csv $(dirname $Q)/config.json $OUT/
gzip -c $(dirname $Q)/responses_$MODEL.jsonl > $OUT/responses_$MODEL.jsonl.gz
gzip -c $Q > $OUT/queue.jsonl.gz
$PY - <<'EOF' > $OUT/HANDOFF.md
import json, pandas as pd
out = "artifacts_committed/phase2_dev"
sel = json.load(open(f"{out}/retrieval_selection_dev.json"))
sc = pd.read_csv(f"{out}/scores_gpt-audio-2025-08-28.csv")
print("# Overnight dev pilot results (fold 0 only; no test data touched)\n")
print("## Retrieval setting selected on dev (kNN macro-F1)\n")
print(pd.DataFrame(sel).T[["center", "top_n", "m", "k", "dev_knn_macro_f1", "same_corpus_rate", "dev_queries"]].to_markdown())
print("\n## gpt-audio pilot: w/o RAG vs w/ RAG vs kNN on the SAME queries\n")
piv = sc.pivot(index="task", columns="condition", values="macro_f1").round(3)
print(piv.to_markdown())
print("\n## Details\n")
cols = [c for c in ["task", "condition", "n_queries", "accuracy", "macro_f1", "macro_f1_ci95", "invalid_rate",
                    "copies_evidence_majority", "acc_when_evidence_right", "acc_when_evidence_wrong"] if c in sc]
print(sc[cols].round(3).to_markdown(index=False))
EOF
echo "=== $(date) done"
