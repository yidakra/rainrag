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
   it, grants access, and pastes back the code. Each URL is single-use: it
   carries a PKCE challenge whose verifier is written next to the token file
   and consumed by the matching ``--auth-code`` run, so a code obtained from
   an older URL cannot be exchanged. The refresh token is stored in
   ``data/google_oauth_token.json`` and never needs repeating.
3. Every later run appends one snapshot per video to
   ``data/youtube_metrics.csv`` in the «YT metrics» sheet's column schema,
   which the Library UI already reads.

Everything that talks to Google is isolated in ``fetch_video_metrics`` so
the transform and the CSV writing are testable without credentials.
"""

from __future__ import annotations

import csv
import fcntl
import os
import re
import time
from pathlib import Path
from typing import Any

from rainrag.library_performance import DISTRIBUTION_COLUMNS, format_distribution


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


def rows_to_dicts(
    column_headers: list[dict[str, Any]], rows: list[list[Any]]
) -> list[dict[str, Any]]:
    """Zip one response's positional rows with *its own* column headers.

    Done per batch, never after merging: the requested metric set can shrink
    mid-pull when the API rejects a metric, and rows from an earlier batch
    zipped against a later, narrower header list would land every value after
    the dropped column under the wrong name.
    """
    names = [h["name"] for h in column_headers]
    if rows and "video" not in names:
        raise ValueError("response has no video dimension")
    return [dict(zip(names, row, strict=True)) for row in rows]


def rows_to_snapshot(records: list[dict[str, Any]], snapshot_date: str) -> list[dict[str, Any]]:
    """Reshape merged per-video records into CSV rows in the sheet's schema.

    Metrics absent from a record stay blank so the reader (``load_metrics``)
    simply sees them as missing.
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        csv_row: dict[str, Any] = dict.fromkeys(CSV_COLUMNS, "")
        csv_row["youtube_id"] = rec["video"]
        csv_row["snapshot_date"] = snapshot_date
        for m in API_METRICS:
            if rec.get(m) is not None:
                csv_row[m] = rec[m]
        for col in DISTRIBUTION_COLUMNS:
            if rec.get(col):
                csv_row[col] = rec[col]
        out.append(csv_row)
    return out


def append_snapshot(path: Path, csv_rows: list[dict[str, Any]]) -> int:
    """Append rows under a flock; header written once. Returns rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Position is fixed at open(), before the lock; a concurrent first
            # run could have written the header in between. Re-check under it.
            f.seek(0, os.SEEK_END)
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


def _write_private(path: Path, data: str) -> None:
    """Write owner-only: these files grant read access to channel revenue."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(data)


def _write_token(token_json: Path, data: str) -> None:
    """Owner-only file: the refresh token grants read access to channel revenue."""
    _write_private(token_json, data)


def _verifier_path(token_json: Path) -> Path:
    """Where the PKCE verifier waits between the --auth and --auth-code runs."""
    return token_json.with_name(token_json.name + ".verifier")


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
        from google.auth.exceptions import RefreshError

        try:
            creds.refresh(Request())
        except RefreshError as exc:
            # The one failure a nightly timer must explain rather than dump.
            # An app left in Testing issues refresh tokens that die after
            # seven days; a revoked grant looks the same from here.
            raise SystemExit(
                "YouTube consent has expired or was revoked (refresh failed: "
                f"{exc}). Re-run with --auth and have the channel owner consent "
                "again, choosing the Library brand account. If this recurs "
                "weekly, the OAuth app is still in Testing: publish it."
            ) from exc
        _write_token(token_json, creds.to_json())
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
    verifier_path = _verifier_path(token_json)
    if auth_code is None:
        url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        # authorization_url() mints a PKCE verifier that fetch_token() has to
        # send back. --auth and --auth-code are separate processes, so an
        # in-memory verifier is gone by the time the code arrives and Google
        # rejects the exchange. Keep it on disk, owner-only, until it is used.
        if flow.code_verifier:
            _write_private(verifier_path, flow.code_verifier)
        raise SystemExit(
            "Open this URL as the channel owner, grant access, then re-run with\n"
            "  --auth-code <the 'code=' value from the address bar after the redirect>\n\n"
            f"{url}"
        )
    saved = verifier_path.read_text(encoding="utf-8").strip() if verifier_path.exists() else ""
    if saved:
        flow.code_verifier = saved
    elif flow.autogenerate_code_verifier:
        raise SystemExit(
            f"No PKCE verifier at {verifier_path}, so this code cannot be exchanged.\n"
            "Re-run with --auth and use the URL it prints: a code obtained from an "
            "earlier URL is tied to a verifier that no longer exists."
        )
    flow.fetch_token(code=auth_code)
    creds = flow.credentials
    _write_token(token_json, creds.to_json())
    verifier_path.unlink(missing_ok=True)
    return creds


def fetch_video_metrics(
    creds: Any, video_ids: list[str], start_date: str, end_date: str
) -> list[dict[str, Any]]:
    """Lifetime-to-date metrics per video, as one dict per video.

    Batches of 500 ids (the API's filter cap). Each batch's rows are zipped
    with that batch's own headers before merging, so a metric the API rejects
    on a later batch cannot shift earlier rows. Rejected metrics are dropped
    for the rest of the pull and the batch retried, so a channel without
    monetisation still yields views and retention.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    service = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
    metrics = list(API_METRICS)
    records: list[dict[str, Any]] = []
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
        records.extend(rows_to_dicts(resp.get("columnHeaders", []), resp.get("rows", [])))
    return records


def _rejected_metric(message: str, metrics: list[str]) -> str | None:
    """The exact requested metric an API error names, if any.

    Matched as a whole identifier, not a substring: the API quotes the bad
    name in parentheses or quotes, and a substring test would let a shorter
    metric name hide inside a longer one.
    """
    for m in metrics:
        if re.search(rf"(?<![A-Za-z]){re.escape(m)}(?![A-Za-z])", message):
            return m
    return None


def marginals(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    """Age and gender marginals from joint (ageGroup, gender, viewerPercentage) rows.

    One request per video returns the joint table; summing it both ways gives
    the two columns the sheet has, and one API call instead of two.
    """
    age: dict[str, float] = {}
    gender: dict[str, float] = {}
    for r in rows:
        pct = float(r.get("viewerPercentage") or 0.0)
        a, g = str(r.get("ageGroup") or ""), str(r.get("gender") or "")
        if a:
            age[a] = age.get(a, 0.0) + pct
        if g:
            gender[g] = gender.get(g, 0.0) + pct
    return age, gender


RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def fetch_video_demographics(
    creds: Any,
    video_ids: list[str],
    start_date: str,
    end_date: str,
    workers: int = 8,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Age and gender mix per video: one request each, by API design.

    The demographics report accepts ``video`` only as a filter, never as a
    dimension, so 239 uploads mean 239 requests, and the API answers each in
    anything from under a second to sixteen. Sequentially that is half an
    hour; measured on the live channel, 2026-09-14. Eight workers bring it
    to a few minutes. httplib2 is not thread-safe, so each worker builds its
    own service object.

    One retry, after a short pause, on a rate limit, a server-side error, or
    a transport failure such as the socket timeout the script sets. Anything
    else counts the video as failed and moves on; nothing escapes the worker. A video with no
    rows is not a failure: YouTube withholds demographics below a views
    threshold, and that video simply gets no cell.

    Returns the formatted cells per video and the ids whose request failed.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    local = threading.local()

    def service() -> Any:
        if not hasattr(local, "svc"):
            local.svc = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
        return local.svc

    def one(vid: str) -> tuple[str, dict[str, str] | None, bool]:
        for attempt in (1, 2):
            try:
                resp = (
                    service()
                    .reports()
                    .query(
                        ids="channel==MINE",
                        startDate=start_date,
                        endDate=end_date,
                        metrics="viewerPercentage",
                        dimensions="ageGroup,gender",
                        filters=f"video=={vid}",
                    )
                    .execute()
                )
                break
            except Exception as exc:  # noqa: BLE001 - one video must not sink the night
                # HttpError carries a status; a socket timeout or any other
                # transport failure does not, and letting it escape the worker
                # aborted the whole pull before the snapshot was written. Both
                # kinds get one retry when they look transient, then count the
                # video as failed and let the rest continue.
                status = getattr(getattr(exc, "resp", None), "status", None)
                transient = status in RETRYABLE_STATUSES or not isinstance(exc, HttpError)
                if attempt == 1 and transient:
                    time.sleep(2.0)
                    continue
                return vid, None, True
        names = [c["name"] for c in resp.get("columnHeaders", [])]
        rows = [dict(zip(names, r, strict=False)) for r in resp.get("rows") or []]
        if not rows:
            return vid, None, False
        age, gender = marginals(rows)
        return (
            vid,
            {
                DISTRIBUTION_COLUMNS[0]: format_distribution(age),
                DISTRIBUTION_COLUMNS[1]: format_distribution(gender),
            },
            False,
        )

    out: dict[str, dict[str, str]] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for vid, cells, bad in pool.map(one, video_ids):
            if bad:
                failed.append(vid)
            elif cells:
                out[vid] = cells
    return out, failed


def merge_demographics(
    records: list[dict[str, Any]], demographics: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """Attach the demographic cells to the metric records they belong to."""
    for rec in records:
        extra = demographics.get(str(rec.get("video")))
        if extra:
            rec.update(extra)
    return records
