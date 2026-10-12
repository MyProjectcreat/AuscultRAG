#!/usr/bin/env bash
# TEST run of the structured-report evaluation (folds 1-4 pooled; every setting frozen on dev):
# wait for test-round retrieval -> prepare queue (200 queries/task) -> run each model -> score -> copy to artifacts_committed/
# Usage: scripts/run_test_reports.sh MODEL [MODEL ...]     (resumable; re-run to continue)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=${AUSCULTRAG_ARTIFACTS:-artifacts}
NAME=report_test_v3
Q=$A/llm/$NAME/queue.jsonl
OUT=artifacts_committed/test_reports
echo "=== $(date) start: models $*"

until grep -q "test retrieval done" $A/logs/test_retrieval.log; do
  if grep -q Traceback $A/logs/test_retrieval.log; then echo "test retrieval failed"; exit 1; fi
  sleep 30
done
[ -f "$Q" ] || $PY scripts/report_prepare.py --rounds 1,2,3,4 --per-task 200 --name $NAME
echo "=== $(date) queue: $(wc -l < $Q) items"

for M in "$@"; do
  ( for attempt in 1 2 3; do $PY scripts/llm_run.py --queue $Q --model $M --workers ${WORKERS:-6}; done ) \
      > $A/logs/${NAME}_$M.log 2>&1 &
done
wait
mkdir -p $OUT
for M in "$@"; do
  $PY scripts/report_score.py --queue $Q --model $M > $OUT/report_scores_$M.txt 2>&1 || true
  cp $A/llm/$NAME/report_scores_$M.csv $A/llm/$NAME/report_field_scores_$M.csv $OUT/ 2>/dev/null || true
  gzip -c $A/llm/$NAME/responses_$M.jsonl > $OUT/responses_$M.jsonl.gz
done
gzip -c $Q > $OUT/queue.jsonl.gz
echo "=== $(date) done"
