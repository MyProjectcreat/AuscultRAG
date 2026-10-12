#!/usr/bin/env bash
# Tier: cpu. Re-acquire Coswara (iiscleap/Coswara-Data, CC BY 4.0) and extract it.
# Usage: scripts/acquire/coswara.sh [RAW_ROOT]   (default: $AUSCULTRAG_RAW or <repo>/raw_data)
set -euo pipefail
RAW_ROOT="${1:-${AUSCULTRAG_RAW:-$(cd "$(dirname "$0")/../.." && pwd)/raw_data}}"
DEST="$RAW_ROOT/coswara"
REPO_URL="https://github.com/iiscleap/Coswara-Data.git"
# Pin the exact upstream commit so re-acquisition is reproducible.
PINNED_COMMIT="${COSWARA_COMMIT:-4942c97e31de7180a93d17f2e7530a9c543cfd50}"  # commit used for the 2026-10 rebuild

mkdir -p "$DEST"
if [ ! -d "$DEST/Coswara-Data/.git" ]; then
  git clone --depth 1 "$REPO_URL" "$DEST/Coswara-Data"
fi
cd "$DEST/Coswara-Data"
if [ -n "$PINNED_COMMIT" ]; then
  git fetch --depth 1 origin "$PINNED_COMMIT" && git checkout -q "$PINNED_COMMIT"
fi
git rev-parse HEAD > "$DEST/UPSTREAM_COMMIT"

# Reassemble split archives per date folder and extract; skip folders already extracted.
mkdir -p "$DEST/Extracted_data"
for d in 20*/; do
  d="${d%/}"
  if [ -f "$DEST/Extracted_data/.done_$d" ]; then continue; fi
  cat "$d"/*.tar.gz.* | tar -xz -C "$DEST/Extracted_data/"
  touch "$DEST/Extracted_data/.done_$d"
done
echo "Coswara extracted under $DEST/Extracted_data (commit $(cat "$DEST/UPSTREAM_COMMIT"))"
