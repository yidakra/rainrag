#!/usr/bin/env bash
set -euo pipefail

# Ensure files created in this script are owner-only by default
umask 077

dir=./secrets
mkdir -p "$dir"
chmod 700 "$dir"

# Ensure secrets/ is ignored in git to avoid committing API key placeholders
if [ -f .gitignore ]; then
  if ! grep -qx "^secrets/$" .gitignore; then
    echo "secrets/" >> .gitignore
    echo "Updated .gitignore with secrets/"
  fi
else
  echo "secrets/" > .gitignore
  echo "Created .gitignore with secrets/"
fi

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
