#!/usr/bin/env bash
set -euo pipefail

EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-/root/rainrag/embeddings}"

if [[ ! -d "$EMBEDDINGS_DIR" ]]; then
  echo "Embeddings directory not found: $EMBEDDINGS_DIR" >&2
  exit 1
fi

: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID}"
: "${R2_ACCESS_KEY_ID:?Set R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?Set R2_SECRET_ACCESS_KEY}"
: "${R2_BUCKET:?Set R2_BUCKET}"

R2_PREFIX="${R2_PREFIX:-embeddings}"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"
export AWS_EC2_METADATA_DISABLED=true

aws s3 sync \
  "$EMBEDDINGS_DIR" \
  "s3://${R2_BUCKET}/${R2_PREFIX}" \
  --endpoint-url "$ENDPOINT" \
  --only-show-errors
