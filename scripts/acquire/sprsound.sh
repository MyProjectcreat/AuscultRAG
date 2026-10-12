#!/usr/bin/env bash
# Tier: cpu. Re-acquire SPRSound (SJTU paediatric respiratory sound database, CC-BY-4.0) from GitHub,
# pinned to a commit. Contains BioCAS2022-2025 releases + patient summary (diagnosis) CSVs.
# Usage: scripts/acquire/sprsound.sh [RAW_ROOT]
set -euo pipefail
RAW_ROOT="${1:-${AUSCULTRAG_RAW:-$(cd "$(dirname "$0")/../.." && pwd)/raw_data}}"
DEST="$RAW_ROOT/sprsound"
PINNED_COMMIT="${SPRSOUND_COMMIT:-bca1e51422a42a042441010081519610ef3845d0}"
mkdir -p "$DEST"
[ -d "$DEST/SPRSound/.git" ] || git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git "$DEST/SPRSound"
cd "$DEST/SPRSound"
if [ -n "$PINNED_COMMIT" ]; then git fetch --depth 1 origin "$PINNED_COMMIT" && git checkout -q "$PINNED_COMMIT"; fi
git rev-parse HEAD > "$DEST/UPSTREAM_COMMIT"
echo "SPRSound at $(cat "$DEST/UPSTREAM_COMMIT")"
