#!/usr/bin/env bash
set -euo pipefail

dir=./secrets
mkdir -p "$dir"
chmod 700 "$dir"

for key in mistral_api_key cohere_api_key openai_api_key anthropic_api_key google_api_key; do
  file="$dir/${key}.txt"
  if [ ! -f "$file" ]; then
    echo "# Placeholder for $key. Replace with your real API key." > "$file"
    echo "Created: $file"
  else
    echo "Exists:  $file"
  fi
  chmod 600 "$file"
done

echo "Done. Edit secrets/*.txt with your real API keys (do not commit)."