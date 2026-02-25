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

Nginx config
- Copy the vhost file from deploy/nginx and enable it in your nginx setup.
- The config assumes TLS certs at:
  /etc/nginx/certs/rag.tvrain.tv.crt
  /etc/nginx/certs/rag.tvrain.tv.key

Environment
- Put secrets in /home/ubuntu/rainrag/.env (RAINRAG_PASSWORD_HASH already added per your note).
- Set MISTRAL_API_KEY or other provider keys as needed.
- Optional: RAINRAG_AUTH_TOKEN for API protection.

Notes
- The Streamlit service sets RAINRAG_API_URL=https://rag.tvrain.tv/api and nginx strips /api before proxying.
- If you want to use /embeddings, either update config.yaml or symlink ./embeddings to /embeddings.
