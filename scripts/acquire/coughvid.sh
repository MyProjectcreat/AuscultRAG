#!/usr/bin/env bash
# Tier: cpu. Re-acquire COUGHVID v3 (Zenodo record 7024894, DOI 10.5281/zenodo.7024894, CC-BY-4.0),
# verify against Zenodo's md5, extract, then transcode webm/ogg to 16-bit PCM wav (native rate) so
# every downstream stage can read it with soundfile.
# Usage: scripts/acquire/coughvid.sh [RAW_ROOT]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../.venv/bin/python"
RAW_ROOT="${1:-${AUSCULTRAG_RAW:-$(cd "$(dirname "$0")/../.." && pwd)/raw_data}}"
DEST="$RAW_ROOT/coughvid"
mkdir -p "$DEST"
"$PY" "$HERE/parallel_download.py" "https://zenodo.org/api/records/7024894/files/public_dataset_v3.zip/content" \
  "$DEST/public_dataset_v3.zip" --md5 7c6cdf184748a2600538c331ba0ba718 --workers 8
if [ ! -f "$DEST/.extracted" ]; then
  unzip -q -o "$DEST/public_dataset_v3.zip" -d "$DEST" && touch "$DEST/.extracted"
fi
cd "$HERE/../.." && "$PY" -m auscultrag.corpora.coughvid_transcode --raw-root "$RAW_ROOT"
