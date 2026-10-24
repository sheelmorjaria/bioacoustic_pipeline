#!/usr/bin/env bash
# Mirror public GCS prefixes via parallel curl (no GIL serialization).
# Usage: mirror_prefixes_curl.sh <prefix> ...
set -u
BUCKET=esp-data-274503
DEST="${ALP_DATA_HOME:-data/alp}"
for PREFIX in "$@"; do
  LIST=$(mktemp)
  TOKEN=""
  : > "$LIST"
  while : ; do
    URL="https://storage.googleapis.com/storage/v1/b/${BUCKET}/o?prefix=${PREFIX//\//%2F}&maxResults=1000"
    [ -n "$TOKEN" ] && URL="${URL}&pageToken=${TOKEN}"
    PAGE=$(curl -s "$URL")
    echo "$PAGE" | grep -o '"name": "[^"]*"' | sed 's/"name": "//; s/"//' >> "$LIST"
    echo "$PAGE" | grep -o '"size": "[0-9]*"' | sed 's/"size": "//; s/"//' > "${LIST}.sizes"
    TOKEN=$(echo "$PAGE" | grep -o '"nextPageToken": "[^"]*"' | sed 's/.*: "//; s/"//')
    [ -z "$TOKEN" ] && break
  done
  N=$(wc -l < "$LIST")
  echo "== $PREFIX: $N objects"
  paste -d'|' "$LIST" "${LIST}.sizes" | while IFS='|' read -r NAME SIZE; do
    echo "$NAME $SIZE"
  done > "${LIST}.pairs"
  cat "${LIST}.pairs" | xargs -P 32 -n 2 bash -c '
    NAME="$0"; SIZE="$1"; TARGET="'"$DEST"'/$NAME"
    if [ -f "$TARGET" ] && [ "$(stat -c%s "$TARGET")" = "$SIZE" ]; then exit 0; fi
    mkdir -p "$(dirname "$TARGET")"
    curl -s --fail --retry 4 -o "$TARGET" "https://storage.googleapis.com/'"$BUCKET"'/$NAME" || echo "FAIL $NAME" >&2
  '
  rm -f "$LIST" "${LIST}.sizes" "${LIST}.pairs"
  echo "== $PREFIX DONE"
done
echo "MIRROR-DONE"
