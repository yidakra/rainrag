Deployment notes (systemd + nginx)

1) Copy unit files
   sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-api.service /etc/systemd/system/
   sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-streamlit.service /etc/systemd/system/

2) Reload systemd
   sudo systemctl daemon-reload

3) Enable and start services
   sudo systemctl enable --now rainrag-api
   sudo systemctl enable --now rainrag-streamlit

4) Check status
   sudo systemctl status rainrag-api
   sudo systemctl status rainrag-streamlit

Hourly incremental updater
1) Ensure incremental mode is enabled in `/home/ubuntu/rainrag/config.yaml`:
   incremental:
     enabled: true

2) Install timer units:
   chmod +x /home/ubuntu/rainrag/deploy/systemd/install_incremental_timer.sh
   /home/ubuntu/rainrag/deploy/systemd/install_incremental_timer.sh

3) Inspect timer + service:
   sudo systemctl status rainrag-incremental-update.timer
   sudo systemctl status rainrag-incremental-update.service
   sudo journalctl -u rainrag-incremental-update.service -n 200 --no-pager

4) Logs:
   /home/ubuntu/rainrag/logs/incremental-hourly.log

Notes:
- The updater script uses a lockfile (`/tmp/rainrag-incremental.lock`) to prevent overlapping runs.
- It refuses to run when `incremental.enabled` is false.
- It performs a manifest sanity check to avoid accidental full rebuilds when manifest state is stale.

Deploy on merge (every two minutes, fetch + fast-forward + restart Streamlit)
1) Install and enable:
   sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-deploy.service /etc/systemd/system/
   sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-deploy.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now rainrag-deploy.timer

2) Details, refusals and what it will not restart: deploy/systemd/DEPLOY_ON_MERGE.md

Daily YouTube Analytics snapshot (05:20 UTC, metrics + age/gender per upload)
1) Needs data/google_oauth_token.json from a one-time consent by the channel owner
   (scripts/youtube_analytics_pull.py --auth prints a consent URL; the channel
   owner opens it, picks the Library brand account in Google's chooser, and
   sends back the address; then scripts/youtube_analytics_pull.py --auth-code
   '<the code= value from that address>').
2) Install and enable:
   sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-analytics.service /etc/systemd/system/
   sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-analytics.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now rainrag-analytics.timer

3) Outcome of every run is posted to #rainrag-test; an expired consent says so
   and how to renew it.

Nginx config
- Copy the vhost file from deploy/nginx and enable it in your nginx setup.
- The config assumes TLS certs at:
  /etc/nginx/certs/rag.tvrain.tv.crt
  /etc/nginx/certs/rag.tvrain.tv.key

Environment
- Put secrets in /home/ubuntu/rainrag/.env (RAINRAG_PASSWORD_HASH already added per your note).
- Set MISTRAL_API_KEY or other provider keys as needed.
- Optional: RAINRAG_AUTH_TOKEN for API protection.
- For external DNS deployments, prefer leaving RAINRAG_ALLOWED_HOSTS and
  RAINRAG_CORS_ORIGINS unset and taking the defaults in `rainrag.api`, which
  already list both `rag.tvrain.io` and `rag.tvrain.tv`. If you do set them,
  include every hostname a browser will use. Pinning them to `.tv` only is what
  made TrustedHostMiddleware reject the gateway's Host and killed media
  playback; those lines were removed from `rainrag-api.service` for that reason.

Notes
- The Streamlit services set RAINRAG_API_URL=http://127.0.0.1:8001. That call is
  made by the Streamlit process itself, so it goes straight to the API rather
  than out through a public name and back. The old value was
  https://rag.tvrain.tv/api, whose public path now returns 522.
- **`.env` wins.** `EnvironmentFile=` is read after the inline `Environment=`
  lines, so a variable set in `/home/ubuntu/rainrag/.env` overrides the unit.
  Change the file, not just the unit line.
- Browser-facing media URLs come from RAINRAG_ASSET_URL, which is separate on
  purpose: leave it unset for same-origin relative URLs, which is correct behind
  a proxy. Never let it inherit a loopback address, or the reader's browser is
  told to fetch video from itself.
- If you want to use /embeddings, either update config.yaml or symlink ./embeddings to /embeddings.
