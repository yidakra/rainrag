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
# fail fast if Qdrant is unresponsive; allow overrides via CURL_CONNECT_TIMEOUT/CURL_MAX_TIME
connect_timeout="${CURL_CONNECT_TIMEOUT:-5}"
max_time="${CURL_MAX_TIME:-30}"
response="$(curl -fsS \
  --connect-timeout "$connect_timeout" \
  -m "$max_time" \
  -X POST "${QDRANT_URL}/collections/${QDRANT_COLLECTION}/snapshots")"

snapshot_name="$(printf '%s' "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["result"]["name"])')"
if [[ -z "$snapshot_name" ]]; then
  echo "Failed to parse snapshot name from Qdrant response" >&2
  exit 1
fi

snapshot_file="${SNAPSHOT_DIR}/${TIMESTAMP}-${snapshot_name}"
# ensure temporary snapshot file is removed on exit (success or failure)
trap 'rm -f "$snapshot_file" >/dev/null 2>&1 || true' EXIT

echo "Downloading snapshot to: $snapshot_file"
# download with connection and overall timeouts to prevent hanging indefinitely
# allow customization via env vars CURL_CONNECT_TIMEOUT and CURL_MAX_TIME
connect_timeout="${CURL_CONNECT_TIMEOUT:-10}"
max_time="${CURL_MAX_TIME:-300}"
# run curl and capture its exit code explicitly
curl -fsS \
  --connect-timeout "$connect_timeout" \
  -m "$max_time" \
  "${QDRANT_URL}/collections/${QDRANT_COLLECTION}/snapshots/${snapshot_name}" \
  -o "$snapshot_file"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Error: failed to download snapshot from Qdrant (exit code $rc)" >&2
  exit $rc
fi

r2_target="s3://${R2_BUCKET}/${QDRANT_PREFIX}/${QDRANT_COLLECTION}/"
echo "Uploading snapshot to: ${r2_target}"

aws s3 cp \
  "$snapshot_file" \
  "${r2_target}$(basename "$snapshot_file")" \
  --endpoint-url "$ENDPOINT" \
  --only-show-errors

echo "Qdrant snapshot backup completed: $(basename "$snapshot_file")"
# remove local snapshot now that it's stored safely
rm -f "$snapshot_file" >/dev/null 2>&1 || true
