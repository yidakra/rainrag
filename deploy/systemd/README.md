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

Nginx config
- Copy the vhost file from deploy/nginx and enable it in your nginx setup.
- The config assumes TLS certs at:
  /etc/nginx/certs/rag.tvrain.tv.crt
  /etc/nginx/certs/rag.tvrain.tv.key

Environment
- Put secrets in /home/ubuntu/rainrag/.env (RAINRAG_PASSWORD_HASH already added per your note).
- Set MISTRAL_API_KEY or other provider keys as needed.
- Optional: RAINRAG_AUTH_TOKEN for API protection.
- For external DNS deployments, set:
  - RAINRAG_ALLOWED_HOSTS=rag.tvrain.tv,localhost,127.0.0.1
  - RAINRAG_CORS_ORIGINS=https://rag.tvrain.tv

Notes
- The Streamlit service sets RAINRAG_API_URL=https://rag.tvrain.tv/api and nginx strips /api before proxying.
- If you want to use /embeddings, either update config.yaml or symlink ./embeddings to /embeddings.
- The alternate Streamlit service (`rainrag-streamlit-ip.service`) also targets https://rag.tvrain.tv/api (no intranet IP dependency).
