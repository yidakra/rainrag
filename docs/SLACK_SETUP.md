# Slack Connector Setup

The Slack connector lets journalists query the RainRAG archive directly from
Slack — no new tool to learn, just the chat interface they already use daily.

Three ways to ask:

- **Mention the bot in a channel**: `@RainRAG когда впервые обсуждали закон об иноагентах?`
  The answer arrives as a threaded reply.
- **DM the bot**: just type a question.
- **Slash command**: `/rainrag what happened during the 2020 protests?`
  The answer is ephemeral (visible only to you) by default.

The answer language follows the question: Cyrillic → Russian, otherwise
English. Date filters can be embedded in any question:
`протесты from:2020-01-01 to:2020-12-31`.

## Architecture

The connector is a small standalone service (port 8002 by default) that
receives Slack webhooks and forwards questions to the existing RainRAG API
(`/query`), exactly like the Streamlit UI does. It has no Slack SDK
dependency — webhook signatures are verified with a stdlib HMAC and answers
are posted with one `chat.postMessage` call via httpx.

```
Slack ──HTTPS──▶ nginx /slack/ ──▶ connector :8002 ──▶ RainRAG API :8001
  ◀── chat.postMessage / response_url ──┘
```

Because it goes through the API, Slack queries respect the same auth,
concurrency limits, timeouts, and show up in the same usage accounting
(`[usage] event=slack_query` journal lines).

## 1. Create the Slack app

1. Go to <https://api.slack.com/apps> → **Create New App** → **From scratch**.
2. Name it (e.g. `RainRAG`) and pick your workspace.
3. Under **OAuth & Permissions**, add these **Bot Token Scopes**:
   - `app_mentions:read` — see mentions in channels
   - `im:history` — read DMs sent to the bot
   - `chat:write` — post answers
4. Click **Install to Workspace** and copy the **Bot User OAuth Token**
   (`xoxb-...`).
5. Under **Basic Information**, copy the **Signing Secret**.

## 2. Configure and start the connector

Add to `.env` (see `.env.example`):

```bash
SLACK_SIGNING_SECRET=<signing secret>
SLACK_BOT_TOKEN=xoxb-...
# RAINRAG_API_URL and RAINRAG_AUTH_TOKEN as for the Streamlit UI
```

Run it:

```bash
# Directly
rainrag slack --host 127.0.0.1 --port 8002

# Or as a service
sudo cp deploy/systemd/rainrag-slack.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rainrag-slack
```

Expose it through nginx — `deploy/nginx/rag.tvrain.tv.conf` already includes
the `/slack/` location block proxying to port 8002.

Check it's alive:

```bash
curl -s http://127.0.0.1:8002/health
# {"status":"ok","signing_secret_configured":true,"bot_token_configured":true,...}
```

## 3. Wire up the webhooks

Back in the Slack app settings:

1. **Event Subscriptions** → enable, set Request URL to
   `https://rag.tvrain.tv/slack/events`. Slack sends a verification challenge;
   the connector answers it automatically, so the URL should turn green.
2. Under **Subscribe to bot events**, add:
   - `app_mention`
   - `message.im`
3. **Slash Commands** → create `/rainrag` with Request URL
   `https://rag.tvrain.tv/slack/commands`.
4. Reinstall the app if Slack prompts you to.

Invite the bot to the channels where journalists work: `/invite @RainRAG`.

## Options

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAINRAG_SLACK_COMMAND_RESPONSE` | `ephemeral` | `/rainrag` answers visible to the asker only, or `in_channel` for everyone |
| `RAINRAG_SLACK_TOP_K` | backend default | Context chunks to retrieve per question |
| `RAINRAG_SLACK_QUERY_TIMEOUT_SECONDS` | `300` | HTTP timeout towards the RainRAG API; keep above the API's own query timeout |

## Security notes

- Every webhook is verified against the Slack signing secret (HMAC-SHA256
  over the raw body) with a 5-minute replay window; unsigned or stale
  requests get 401.
- Retried/redelivered events are deduplicated by `event_id`, and bot messages
  (including the connector's own posts) are ignored, so the bot cannot loop
  on itself.
- Source links in answers use public `web_url` metadata only. Internal media
  URLs are never posted, because they would need the archive auth token
  embedded in them.
- The connector holds no archive credentials beyond the API bearer token it
  already needs; a compromised Slack workspace cannot reach media files.

## Troubleshooting

- **URL verification fails**: the connector isn't reachable from the internet
  or `SLACK_SIGNING_SECRET` doesn't match. Check `journalctl -u rainrag-slack`.
- **Bot acks but never answers**: `SLACK_BOT_TOKEN` missing/invalid, or the
  RainRAG API is down — both are logged. Check `/health` on both services.
- **`not_in_channel` errors in logs**: invite the bot to the channel.
- **Answers time out**: the API's `RAINRAG_QUERY_TIMEOUT_SECONDS` applies;
  the connector reports the failure to the user rather than hanging.
