#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/rainrag}"
CONFIG_PATH="${RAINRAG_CONFIG:-$REPO_DIR/config.yaml}"
LOCK_FILE="${LOCK_FILE:-/tmp/rainrag-incremental.lock}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"

# Safety knobs (override via systemd Environment= or shell env).
BOOTSTRAP_ON_MISSING="${BOOTSTRAP_ON_MISSING:-0}"
MANIFEST_SANITY_MIN_ENTRIES="${MANIFEST_SANITY_MIN_ENTRIES:-100}"
MANIFEST_SANITY_DOCS_THRESHOLD="${MANIFEST_SANITY_DOCS_THRESHOLD:-10000}"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if ! command -v uv >/dev/null 2>&1; then
  log "ERROR: uv is not installed or not in PATH"
  exit 127
fi

cd "$REPO_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another incremental run is already active (lock: $LOCK_FILE). Exiting."
  exit 0
fi

readarray -t CFG < <(
  uv run python - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import json
import sys

import yaml

config_path = Path(sys.argv[1]).resolve()
root = config_path.parent
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

def norm(path_str: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (root / p).resolve()
    return str(p)

incremental_enabled = bool(cfg.get("incremental", {}).get("enabled", False))
manifest_path = norm(cfg.get("incremental", {}).get("manifest_path", "./data/manifest.json"))
docs_output = norm(cfg.get("paths", {}).get("docs_output", "./data/docs.jsonl"))
embeddings_cache = norm(cfg.get("paths", {}).get("embeddings_cache", "./embeddings"))

print("incremental_enabled=" + ("1" if incremental_enabled else "0"))
print("manifest_path=" + manifest_path)
print("docs_output=" + docs_output)
print("embeddings_cache=" + embeddings_cache)
PY
)

for line in "${CFG[@]}"; do
  IFS='=' read -r key value <<< "$line"
    incremental_enabled) incremental_enabled="$value" ;;
    manifest_path) manifest_path="$value" ;;
    docs_output) docs_output="$value" ;;
    embeddings_cache) embeddings_cache="$value" ;;
  esac
done

if [[ "${incremental_enabled:-0}" != "1" ]]; then
  log "ERROR: incremental.enabled is false in $CONFIG_PATH; refusing scheduled run."
  exit 2
fi

manifest_ok=0
if [[ -f "$manifest_path" ]]; then
  manifest_ok=1
fi

docs_ok=0
if [[ -f "$docs_output" ]]; then
  docs_ok=1
fi

cache_ok=0
if [[ -f "$embeddings_cache/embeddings.npy" && -f "$embeddings_cache/metadata.jsonl" ]]; then
  cache_ok=1
fi

if [[ "$manifest_ok" != "1" || "$docs_ok" != "1" || "$cache_ok" != "1" ]]; then
  if [[ "$BOOTSTRAP_ON_MISSING" == "1" ]]; then
    log "Baseline artifacts missing; running one-time FULL pipeline bootstrap."
    uv run rainrag pipeline --config "$CONFIG_PATH"
    log "Full bootstrap completed."
    exit 0
  fi
  log "ERROR: baseline artifacts missing (manifest/docs/cache)."
  log "manifest=$manifest_path exists=$manifest_ok docs=$docs_output exists=$docs_ok cache_dir=$embeddings_cache exists=$cache_ok"
  log "Run a one-time full pipeline first, or set BOOTSTRAP_ON_MISSING=1 for this service."
  exit 3
fi

manifest_entries="$(uv run python - "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    obj = json.loads(p.read_text(encoding="utf-8"))
    print(len(obj) if isinstance(obj, (dict, list)) else 0)
except Exception:
    print(0)
PY
)"

docs_lines="$( { wc -l < "$docs_output"; } 2>/dev/null || echo 0 )"

if [[ "$docs_lines" -ge "$MANIFEST_SANITY_DOCS_THRESHOLD" && "$manifest_entries" -lt "$MANIFEST_SANITY_MIN_ENTRIES" ]]; then
  log "ERROR: manifest sanity check failed."
  log "docs lines=$docs_lines but manifest entries=$manifest_entries (min expected $MANIFEST_SANITY_MIN_ENTRIES)."
  log "Refusing run to avoid accidental full rebuild in incremental mode."
  exit 4
fi

log "Starting incremental pipeline run."
log "config=$CONFIG_PATH manifest_entries=$manifest_entries docs_lines=$docs_lines"
log "Step 1/3: incremental ingestion"
uv run rainrag ingest --config "$CONFIG_PATH" --incremental

log "Step 2/3: incremental embedding"
uv run rainrag embed --config "$CONFIG_PATH" --incremental

log "Step 3/3: incremental indexing"
uv run rainrag index --config "$CONFIG_PATH" --incremental

log "Incremental run completed successfully."
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$LOG_DIR/incremental.last_success"
