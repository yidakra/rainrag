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
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
SNAPSHOT_NAME="${QDRANT_SNAPSHOT_NAME:-}"

mkdir -p "$SNAPSHOT_DIR"

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"
export AWS_EC2_METADATA_DISABLED=true

prefix="s3://${R2_BUCKET}/${QDRANT_PREFIX}/${QDRANT_COLLECTION}/"

if [[ -z "$SNAPSHOT_NAME" ]]; then
  echo "Resolving latest snapshot in ${prefix}"
  SNAPSHOT_NAME="$(aws s3 ls "$prefix" --endpoint-url "$ENDPOINT" | awk '{print $4}' | sort | tail -n 1)"
fi

if [[ -z "$SNAPSHOT_NAME" ]]; then
  echo "No snapshot found at ${prefix}" >&2
  exit 1
fi

local_snapshot="${SNAPSHOT_DIR}/${SNAPSHOT_NAME}"

echo "Downloading snapshot ${SNAPSHOT_NAME} to ${local_snapshot}"
aws s3 cp \
  "${prefix}${SNAPSHOT_NAME}" \
  "$local_snapshot" \
  --endpoint-url "$ENDPOINT" \
  --only-show-errors

echo "Uploading snapshot to Qdrant collection: ${QDRANT_COLLECTION}"
curl -fsS -X POST \
  -F "snapshot=@${local_snapshot}" \
  "${QDRANT_URL}/collections/${QDRANT_COLLECTION}/snapshots/upload" >/dev/null

echo "Qdrant snapshot restore completed: ${SNAPSHOT_NAME}"
