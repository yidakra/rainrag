# Slack Connector Setup

The Slack connector gives journalists the full RainRAG feature set directly
in Slack — no new tool to learn, just the chat interface they already use
daily. It is at feature parity with the Streamlit web UI.

## What journalists can do

Three ways to talk to the bot:

- **Mention it in a channel**: `@RainRAG когда впервые обсуждали закон об иноагентах?`
  The answer arrives as a threaded reply.
- **DM it**: just type a question.
- **Slash command**: `/rainrag what happened during the 2020 protests?`
  The answer is ephemeral (visible only to you) by default.

The answer language follows the question: Cyrillic → Russian, otherwise
English (`lang:ru` / `lang:en` overrides).

### Command reference

| Input | What happens |
| --- | --- |
| `<question>` | Archive Q&A: answer with sources, then transcript excerpts with timecodes, scores and media links |
| `... from:2021-01-01 to:2021-12-31` | Date-range filter (the web UI's date pickers) |
| `... top:10` | Retrieval depth override, 1–20 (the web UI's slider) |
| `... lang:en` | Answer-language override (the web UI's language toggle) |
| `name: вечернее шоу` | Search videos by title (the web UI's name-search mode) |
| `video: <ссылка>` | Import a video by URL, transcribe it, then answer questions about it in that thread (the web UI's upload mode) |
| *(attach a video file in a DM)* | Same as `video:`, from a file instead of a URL |
| `status` | Backend health summary (the web UI's sidebar info) |
| `help` | Command reference |

Every answer's context message carries **"Похожее / Related" buttons** — the
web UI's related-chunks explorer — and, when media links are configured, each
excerpt links the **video at the right timecode** and its **VTT subtitles**
using expiring signed tokens. The top match's actual footage is additionally
posted **as an inline clip** that plays right in Slack (see Options).

## Architecture

The connector is a small standalone service (port 8002 by default) that
receives Slack webhooks and forwards questions to the existing RainRAG API,
exactly like the Streamlit UI does. It has no Slack SDK dependency — webhook
signatures are verified with a stdlib HMAC and replies are posted with plain
httpx calls.

```text
Slack ──HTTPS──▶ nginx /slack/ ──▶ connector :8002 ──▶ RainRAG API :8001
  ◀── chat.postMessage / files upload / response_url ──┘
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
   - `files:read` — download video files journalists attach for import
   - `files:write` — post video clips of retrieved footage into threads
4. Click **Install to Workspace** and copy the **Bot User OAuth Token**
   (`xoxb-...`).
5. Under **Basic Information**, copy the **Signing Secret**.

## 2. Configure and start the connector

Add to `.env` (see `.env.example`):

```bash
SLACK_SIGNING_SECRET=<signing secret>
SLACK_BOT_TOKEN=xoxb-...
# RAINRAG_API_URL and RAINRAG_AUTH_TOKEN as for the Streamlit UI
# RAINRAG_ASSET_URL=https://rag.tvrain.tv   # enables media links in answers
```

Run it:

```bash
# Directly
rainrag slack --host 127.0.0.1 --port 8002

# Or as a service. The unit assumes the ubuntu user and /home/ubuntu/rainrag
# (same convention as rainrag-api.service) -- edit User/Group and the paths
# to match your deployment before enabling.
sudo cp deploy/systemd/rainrag-slack.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rainrag-slack
```

Expose it through nginx by adding the `/slack/` location block from
`deploy/nginx/slack-location.conf` to the existing `server { listen 443 ... }`
block for `rag.tvrain.tv` (in `/etc/nginx/`), then `sudo nginx -t && sudo
systemctl reload nginx`. The live server config is deliberately not kept in
the repo — cert paths and the internal-IP vhost differ per deployment, and a
stale full copy is a footgun.

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
3. **Interactivity & Shortcuts** → enable, set Request URL to
   `https://rag.tvrain.tv/slack/interactive` (powers the "Related" buttons).
4. **Slash Commands** → create `/rainrag` with Request URL
   `https://rag.tvrain.tv/slack/commands`.
5. Reinstall the app if Slack prompts you to.

Invite the bot to the channels where journalists work: `/invite @RainRAG`.

## Options

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAINRAG_ASSET_URL` | *(unset)* | Public base URL for video/VTT links in answers; unset = no media links |
| `RAINRAG_SLACK_CLIP_CHUNKS` | `1` | How many top matches get their footage posted as an inline clip; `0` disables |
| `RAINRAG_SLACK_CLIP_MAX_MB` | `50` | Skip clips larger than this |
| `RAINRAG_SLACK_SHOW_CONTEXT` | `true` | Post the transcript-excerpts message after each answer |
| `RAINRAG_SLACK_COMMAND_RESPONSE` | `ephemeral` | `/rainrag` answers visible to the asker only, or `in_channel` for everyone |
| `RAINRAG_SLACK_TOP_K` | backend default | Context chunks to retrieve per question (users can override with `top:N`) |
| `RAINRAG_SLACK_QUERY_TIMEOUT_SECONDS` | `300` | HTTP timeout towards the RainRAG API; keep above the API's own query timeout |
| `RAINRAG_SLACK_IMPORT_TIMEOUT_SECONDS` | `1800` | Timeout for `video:` URL imports (download happens in-request) |
| `RAINRAG_SLACK_UPLOAD_TIMEOUT_SECONDS` | `1200` | Timeout for attached-file uploads |
| `RAINRAG_SLACK_SESSION_WAIT_SECONDS` | `3600` | How long to wait for transcription before giving up |
| `RAINRAG_SLACK_MAX_UPLOAD_MB` | `512` | Size cap for attached video files |

Inline clips are cut server-side by the API's `/video-clip` endpoint
(ffmpeg stream copy, cached; span capped by `RAINRAG_CLIP_MAX_SECONDS`,
default 300 s) and uploaded to Slack, where they play inline. The message
also links the full video at the right timecode for anything beyond the
excerpt.

## Security notes

- Every webhook (events, commands, interactivity) is verified against the
  Slack signing secret (HMAC-SHA256 over the raw body) with a 5-minute replay
  window; unsigned or stale requests get 401.
- Retried/redelivered events are deduplicated by `event_id`, and bot messages
  (including the connector's own posts) are ignored, so the bot cannot loop
  on itself.
- Media links use short-lived signed tokens (`RAINRAG_MEDIA_TOKEN_TTL_SECONDS`,
  default 12 h) — a link pasted into Slack stops working on its own. The
  standing API secret never appears in a Slack message.
- The connector holds no archive credentials beyond the API bearer token it
  already needs.

## Troubleshooting

- **URL verification fails**: the connector isn't reachable from the internet
  or `SLACK_SIGNING_SECRET` doesn't match. Check `journalctl -u rainrag-slack`.
- **Bot acks but never answers**: `SLACK_BOT_TOKEN` missing/invalid, or the
  RainRAG API is down — both are logged. Check `/health` on both services.
- **`not_in_channel` errors in logs**: invite the bot to the channel.
- **"Related" buttons do nothing**: Interactivity isn't enabled or its
  Request URL is wrong.
- **No media links or clips in answers**: set `RAINRAG_ASSET_URL`; clips also
  need `files:write` and video serving enabled on the API.
- **Answers time out**: the API's `RAINRAG_QUERY_TIMEOUT_SECONDS` applies;
  the connector reports the failure to the user rather than hanging.
