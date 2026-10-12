#!/usr/bin/env bash
# Tier: cpu. Re-acquire the ICBHI 2017 Respiratory Sound Database from the challenge organisers.
# NOTE: the organisers' TLS certificate fails verification, so curl runs with -k. To compensate,
# SHA256 of every downloaded file is recorded on first download and verified on every re-run.
# Usage: scripts/acquire/icbhi.sh [RAW_ROOT]
set -euo pipefail
RAW_ROOT="${1:-${AUSCULTRAG_RAW:-$(cd "$(dirname "$0")/../.." && pwd)/raw_data}}"
DEST="$RAW_ROOT/icbhi"
BASE="https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database"
FILES=(ICBHI_final_database.zip ICBHI_Challenge_diagnosis.txt ICBHI_Challenge_demographic_information.txt
       ICBHI_challenge_train_test.txt filename_differences.txt events.zip)
SUMS="$(cd "$(dirname "$0")" && pwd)/checksums/icbhi.sha256"

mkdir -p "$DEST" "$(dirname "$SUMS")"
cd "$DEST"
for f in "${FILES[@]}"; do
  curl -k -fsSL -C - --retry 5 -o "$f" "$BASE/$f"
done
if [ -f "$SUMS" ]; then
  sha256sum --quiet -c "$SUMS" && echo "ICBHI checksums OK"
else
  sha256sum "${FILES[@]}" > "$SUMS" && echo "Recorded checksums to $SUMS (commit this file)"
fi
if [ ! -f .extracted ]; then
  unzip -q -o ICBHI_final_database.zip && unzip -q -o events.zip -d events && touch .extracted
fi
