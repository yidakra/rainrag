#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/rainrag}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/logs/manifest-backfill.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/rainrag-incremental.lock}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mnt/vod/srv/storage/transcoded}"

mkdir -p "$(dirname "$LOG_FILE")"
cd "$REPO_DIR"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] waiting for lock $LOCK_FILE" | tee -a "$LOG_FILE"

if command -v ionice >/dev/null 2>&1 && command -v nice >/dev/null 2>&1; then
  ionice -c3 nice -n 15 flock "$LOCK_FILE" uv run python scripts/backfill_manifest_doc_ids.py \
    --docs data/docs.jsonl \
    --manifest data/manifest.json \
    --archive-root "$ARCHIVE_ROOT" 2>&1 | tee -a "$LOG_FILE"
else
  flock "$LOCK_FILE" uv run python scripts/backfill_manifest_doc_ids.py \
    --docs data/docs.jsonl \
    --manifest data/manifest.json \
    --archive-root "$ARCHIVE_ROOT" 2>&1 | tee -a "$LOG_FILE"
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] manifest backfill finished" | tee -a "$LOG_FILE"
