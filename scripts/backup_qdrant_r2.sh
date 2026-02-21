#!/usr/bin/env bash
set -euo pipefail

: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID}"
: "${R2_ACCESS_KEY_ID:?Set R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?Set R2_SECRET_ACCESS_KEY}"
: "${R2_BUCKET:?Set R2_BUCKET}"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-broadcast_transcripts}"
QDRANT_PREFIX="${R2_QDRANT_PREFIX:-qdrant}"
SNAPSHOT_DIR="${QDRANT_SNAPSHOT_DIR:-/tmp/rainrag_qdrant_snapshots}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

mkdir -p "$SNAPSHOT_DIR"

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"
export AWS_EC2_METADATA_DISABLED=true

echo "Creating Qdrant snapshot for collection: $QDRANT_COLLECTION"
response="$(curl -fsS -X POST "${QDRANT_URL}/collections/${QDRANT_COLLECTION}/snapshots")"

snapshot_name="$(printf '%s' "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["result"]["name"])')"
if [[ -z "$snapshot_name" ]]; then
  echo "Failed to parse snapshot name from Qdrant response" >&2
  exit 1
fi

snapshot_file="${SNAPSHOT_DIR}/${TIMESTAMP}-${snapshot_name}"

echo "Downloading snapshot to: $snapshot_file"
curl -fsS "${QDRANT_URL}/collections/${QDRANT_COLLECTION}/snapshots/${snapshot_name}" -o "$snapshot_file"

r2_target="s3://${R2_BUCKET}/${QDRANT_PREFIX}/${QDRANT_COLLECTION}/"
echo "Uploading snapshot to: ${r2_target}"

aws s3 cp \
  "$snapshot_file" \
  "${r2_target}$(basename "$snapshot_file")" \
  --endpoint-url "$ENDPOINT" \
  --only-show-errors

echo "Qdrant snapshot backup completed: $(basename "$snapshot_file")"
