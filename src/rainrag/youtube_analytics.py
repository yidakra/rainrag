"""Pull per-video performance from the YouTube Analytics API into the metrics file.

Varya's second query («топ-10 по playbackBasedCpm») needs owner-private
analytics. Public Data API view counts already flow into the map; revenue,
CPM and retention do not, because they require OAuth as the channel owner.
She asked for these to be pulled automatically rather than exported from
Studio by hand, so this module does the API side; the OAuth part is a
one-time setup:

1. In Google Cloud (project ``rainsheets``, where the YouTube APIs are
   already enabled) create an OAuth client of type *Desktop app* and save
   its JSON as ``data/google_oauth_client.json``.
2. Run ``scripts/youtube_analytics_pull.py --auth``: it prints a consent URL.
   Whoever owns (or manages, with revenue access) the Library channel opens
   it, grants access, and pastes back the code. The refresh token is stored
   in ``data/google_oauth_token.json`` and never needs repeating.
3. Every later run appends one snapshot per video to
   ``data/youtube_metrics.csv`` in the «YT metrics» sheet's column schema,
   which the Library UI already reads.

Everything that talks to Google is isolated in ``fetch_video_metrics`` so
the transform and the CSV writing are testable without credentials.
"""

from __future__ import annotations

import csv
import fcntl
from pathlib import Path
from typing import Any


SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
]

# Column names follow the «YT metrics» sheet exactly. The demographic
# viewerPercentage columns need one dimension query per video and are left
# blank in this first pass; everything else is one query per 500 videos.
CSV_COLUMNS = [
    "youtube_id",
    "snapshot_date",
    "views",
    "engagedViews",
    "viewerPercentage: ageGroup",
    "viewerPercentage: gender",
    "viewerPercentage: country",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "subscribersGained",
    "subscribersLost",
    "estimatedRevenue",
    "cpm",
    "playbackBasedCpm",
]

# Requested from the API, most valuable first. engagedViews is newer and some
# channels reject it, so the request is retried without any metric the API
# refuses rather than failing the whole pull.
API_METRICS = [
    "views",
    "engagedViews",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "subscribersGained",
    "subscribersLost",
    "estimatedRevenue",
    "cpm",
    "playbackBasedCpm",
]

FILTER_BATCH = 500  # the API caps a video== filter at 500 ids


def rows_to_snapshot(
    column_headers: list[dict[str, Any]],
    rows: list[list[Any]],
    snapshot_date: str,
) -> list[dict[str, Any]]:
    """Reshape one Analytics response into CSV rows in the sheet's schema.

    The API returns positional rows described by ``columnHeaders``; the
    ``video`` dimension is the id. Metrics the API did not return stay blank
    so the reader (``load_metrics``) simply sees them as absent.
    """
    names = [h["name"] for h in column_headers]
    if "video" not in names:
        raise ValueError("response has no video dimension")
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(zip(names, row, strict=False))
        csv_row: dict[str, Any] = dict.fromkeys(CSV_COLUMNS, "")
        csv_row["youtube_id"] = rec["video"]
        csv_row["snapshot_date"] = snapshot_date
        for m in API_METRICS:
            if m in rec and rec[m] is not None:
                csv_row[m] = rec[m]
        out.append(csv_row)
    return out


def append_snapshot(path: Path, csv_rows: list[dict[str, Any]]) -> int:
    """Append rows under a flock; header written once. Returns rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if f.tell() == 0:
                writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return len(csv_rows)


def chunked(items: list[str], size: int = FILTER_BATCH) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ------------------------------------------------------------------ Google side


def load_credentials(client_json: Path, token_json: Path, auth_code: str | None = None) -> Any:
    """Return authorised credentials, refreshing or completing consent as needed.

    Headless server, so the browser step happens on the user's machine: with
    ``--auth`` the consent URL is printed, the user grants access and pastes
    the code back with ``--auth-code``.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if token_json.exists():
        creds = Credentials.from_authorized_user_file(str(token_json), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_json.write_text(creds.to_json(), encoding="utf-8")
        return creds
    if not client_json.exists():
        raise SystemExit(
            f"OAuth client not found: {client_json}\n"
            "Create a Desktop-app OAuth client in Google Cloud (project rainsheets) and save its JSON there."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(client_json), SCOPES)
    flow.redirect_uri = (
        "http://localhost:1/"  # never reached; the user copies the code from the URL
    )
    if auth_code is None:
        url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        raise SystemExit(
            "Open this URL as the channel owner, grant access, then re-run with\n"
            "  --auth-code <the 'code=' value from the address bar after the redirect>\n\n"
            f"{url}"
        )
    flow.fetch_token(code=auth_code)
    creds = flow.credentials
    token_json.write_text(creds.to_json(), encoding="utf-8")
    return creds


def fetch_video_metrics(
    creds: Any, video_ids: list[str], start_date: str, end_date: str
) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    """Lifetime-to-date metrics per video, in batches of 500 ids.

    Returns (columnHeaders, rows) merged across batches. Metrics the API
    rejects for this channel are dropped and the batch retried, so a channel
    without monetisation still gets views and retention.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    service = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
    metrics = list(API_METRICS)
    headers: list[dict[str, Any]] = []
    rows: list[list[Any]] = []
    for batch in chunked(video_ids):
        while True:
            try:
                resp = (
                    service.reports()
                    .query(
                        ids="channel==MINE",
                        startDate=start_date,
                        endDate=end_date,
                        metrics=",".join(metrics),
                        dimensions="video",
                        filters="video==" + ",".join(batch),
                        maxResults=FILTER_BATCH,
                    )
                    .execute()
                )
                break
            except HttpError as exc:
                bad = _rejected_metric(str(exc), metrics)
                if bad is None:
                    raise
                metrics.remove(bad)
        headers = resp.get("columnHeaders", headers)
        rows.extend(resp.get("rows", []))
    return headers, rows


def _rejected_metric(message: str, metrics: list[str]) -> str | None:
    """Which requested metric an API error complains about, if any."""
    for m in metrics:
        if m in message:
            return m
    return None
