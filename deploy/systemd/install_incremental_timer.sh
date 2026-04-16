#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/rainrag}"
SYSTEMD_DIR="/etc/systemd/system"

sudo cp "$REPO_DIR/deploy/systemd/rainrag-incremental-update.service" "$SYSTEMD_DIR/"
sudo cp "$REPO_DIR/deploy/systemd/rainrag-incremental-update.timer" "$SYSTEMD_DIR/"

sudo systemctl daemon-reload
sudo systemctl enable --now rainrag-incremental-update.timer

echo "Installed and started rainrag-incremental-update.timer"
sudo systemctl list-timers --all | grep -E 'rainrag-incremental-update|NEXT|LAST' || true
