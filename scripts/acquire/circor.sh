#!/usr/bin/env bash
# Tier: cpu. Re-acquire the CirCor DigiScope Phonocardiogram Dataset v1.0.3 (PhysioNet, ODC-By 1.0).
# Source: PhysioNet's open-data S3 mirror (same content as physionet.org, which is much slower here).
# Every file is verified against the project's published SHA256SUMS.txt. Resumable.
# Usage: scripts/acquire/circor.sh [RAW_ROOT] [PARALLEL]
set -euo pipefail
RAW_ROOT="${1:-${AUSCULTRAG_RAW:-$(cd "$(dirname "$0")/../.." && pwd)/raw_data}}"
PARALLEL="${2:-16}"
BASE="https://physionet-open.s3.amazonaws.com/circor-heart-sound/1.0.3"
DEST="$RAW_ROOT/circor/circor-heart-sound-1.0.3"

mkdir -p "$DEST"
cd "$DEST"
curl -fsS --retry 5 -o SHA256SUMS.txt "$BASE/SHA256SUMS.txt"
# Download any file that is missing or fails its checksum.
awk '{print $2}' SHA256SUMS.txt | while read -r f; do
  if [ ! -f "$f" ] || ! grep -F "  $f" SHA256SUMS.txt | sha256sum --quiet -c - >/dev/null 2>&1; then echo "$f"; fi
done | xargs -r -P "$PARALLEL" -I{} sh -c 'mkdir -p "$(dirname "{}")" && curl -fsS --retry 5 -o "{}" "'"$BASE"'/{}"'
sha256sum --quiet -c SHA256SUMS.txt && echo "CirCor: all $(wc -l < SHA256SUMS.txt) files verified"
