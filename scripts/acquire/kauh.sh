#!/usr/bin/env bash
# Tier: cpu. Re-acquire the KAUH lung-sound dataset (Fraiwan et al., Mendeley Data
# DOI 10.17632/jwyy9np4gv.3, CC BY 4.0): chest-wall recordings from 112 patients with diagnoses.
# SHA256 of each file is recorded on first download and verified on re-runs.
# Usage: scripts/acquire/kauh.sh [RAW_ROOT]
set -euo pipefail
RAW_ROOT="${1:-${AUSCULTRAG_RAW:-$(cd "$(dirname "$0")/../.." && pwd)/raw_data}}"
DEST="$RAW_ROOT/kauh"
BASE="https://data.mendeley.com/public-files/datasets/jwyy9np4gv/files"
SUMS="$(cd "$(dirname "$0")" && pwd)/checksums/kauh.sha256"
mkdir -p "$DEST" && cd "$DEST"
curl -fsSL --retry 5 -o "Audio Files.zip" "$BASE/99d7bd63-eb5d-4000-9c2a-a8dd31168cbc/file_downloaded"
curl -fsSL --retry 5 -o "Data annotation.xlsx" "$BASE/bd290409-a700-4c03-ac0f-528c81d8e5c4/file_downloaded"
if [ -f "$SUMS" ]; then sha256sum --quiet -c "$SUMS" && echo "KAUH checksums OK"
else sha256sum "Audio Files.zip" "Data annotation.xlsx" > "$SUMS"; echo "Recorded checksums to $SUMS"; fi
[ -f .extracted ] || { unzip -q -o "Audio Files.zip" -d audio && touch .extracted; }
