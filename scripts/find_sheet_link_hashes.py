#!/usr/bin/env python3
"""Read links from a Google Sheet and map them to video hashes.

Defaults to scanning column H across all data rows in a sheet tab.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


def _column_to_index(column: str) -> int:
    col = column.strip().upper()
    if not col or not col.isalpha():
        raise ValueError(f"Invalid column: {column!r}")
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    # Keep scheme/host/path only for resilient matching.
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )
    return normalized


def _parse_sheet_id(sheet_url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        raise ValueError("Could not parse spreadsheet id from URL")
    return m.group(1)


def _parse_gid(sheet_url: str) -> str:
    parsed = urlparse(sheet_url)
    q = parse_qs(parsed.query)
    if "gid" in q and q["gid"]:
        return q["gid"][0]
    if parsed.fragment.startswith("gid="):
        return parsed.fragment.split("=", 1)[1]
    return "0"


def _fetch_sheet_csv(sheet_url: str, gid: str | None) -> list[list[str]]:
    sheet_id = _parse_sheet_id(sheet_url)
    final_gid = gid if gid is not None else _parse_gid(sheet_url)
    urls = [
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={final_gid}",
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={final_gid}",
    ]

    last_error: Exception | None = None
    for csv_url in urls:
        req = Request(csv_url, headers={"User-Agent": "rainrag-sheet-hash-lookup/1.0"})
        try:
            with urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
            return list(csv.reader(io.StringIO(body)))
        except HTTPError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Could not download sheet as CSV. Ensure the sheet is shared as "
        "'Anyone with the link can view' (or published), then retry."
    ) from last_error


def _http_get_json(url: str, *, access_token: str) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": "rainrag-sheet-hash-lookup/1.0",
        },
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _mint_access_token_from_service_account(
    service_account_file: str, *, writable_sheets: bool = False
) -> str:
    """Mint an OAuth access token from a service-account JSON key file."""
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "google-auth is required for service-account mode. "
            "Install it with: uv add google-auth"
        ) from exc

    if writable_sheets:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
    else:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
    creds = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=scopes,
    )
    creds.refresh(GoogleAuthRequest())
    if not creds.token:
        raise RuntimeError("Failed to mint access token from service account credentials")
    return str(creds.token)


def _resolve_sheet_title(spreadsheet_id: str, gid: str, access_token: str) -> str:
    fields = "sheets.properties(sheetId,title)"
    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
        f"?{urlencode({'fields': fields})}"
    )
    data = _http_get_json(url, access_token=access_token)
    try:
        gid_int = int(gid)
    except ValueError as exc:
        raise RuntimeError(f"Invalid gid: {gid!r}") from exc

    for sheet in data.get("sheets", []):
        props = sheet.get("properties", {})
        if props.get("sheetId") == gid_int:
            title = (props.get("title") or "").strip()
            if title:
                return title
    raise RuntimeError(f"Could not map gid={gid} to a sheet title via Sheets API")


def _fetch_sheet_values_private(
    sheet_url: str,
    gid: str | None,
    *,
    access_token: str,
    column: str,
    row: int | None,
    sheet_name: str | None,
) -> list[tuple[int, str]]:
    spreadsheet_id = _parse_sheet_id(sheet_url)
    final_gid = gid if gid is not None else _parse_gid(sheet_url)
    title = (sheet_name or "").strip() or _resolve_sheet_title(spreadsheet_id, final_gid, access_token)

    if row is not None:
        range_a1 = f"'{title}'!{column.upper()}{row}"
    else:
        # Large bounded range keeps stable 1-based row mapping from row 1.
        range_a1 = f"'{title}'!{column.upper()}1:{column.upper()}1000000"

    encoded_range = quote(range_a1, safe="!:$'")
    query = urlencode({"majorDimension": "ROWS"})
    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/"
        f"{encoded_range}?{query}"
    )
    data = _http_get_json(url, access_token=access_token)
    values = data.get("values", [])

    if row is not None:
        if values and values[0]:
            return [(row, str(values[0][0]).strip())]
        return []

    out: list[tuple[int, str]] = []
    for i, cells in enumerate(values, start=1):
        if not cells:
            continue
        value = str(cells[0]).strip()
        if value:
            out.append((i, value))
    return out


def _write_hashes_to_sheet(
    *,
    spreadsheet_id: str,
    sheet_title: str,
    access_token: str,
    target_column: str,
    row_to_hash: dict[int, str],
) -> int:
    if not row_to_hash:
        return 0

    data = []
    for row_num in sorted(row_to_hash):
        cell_range = f"'{sheet_title}'!{target_column.upper()}{row_num}"
        data.append({"range": cell_range, "values": [[row_to_hash[row_num]]]})

    body = json.dumps({"valueInputOption": "RAW", "data": data}).encode("utf-8")
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values:batchUpdate"
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "rainrag-sheet-hash-lookup/1.0",
        },
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return int(payload.get("totalUpdatedCells", 0))


def _iter_links(rows: list[list[str]], column: str, row: int | None) -> list[tuple[int, str]]:
    col_idx = _column_to_index(column)
    out: list[tuple[int, str]] = []

    if row is not None:
        row_idx = row - 1
        if row_idx < 0 or row_idx >= len(rows):
            return out
        value = rows[row_idx][col_idx].strip() if col_idx < len(rows[row_idx]) else ""
        if value:
            out.append((row, value))
        return out

    for i, cells in enumerate(rows, start=1):
        if col_idx >= len(cells):
            continue
        value = cells[col_idx].strip()
        if value:
            out.append((i, value))
    return out


def _build_url_index(metadata_path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    if not metadata_path.exists():
        return index

    for metadata_file in metadata_path.glob("*.json"):
        if not metadata_file.is_file():
            continue
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_url = (data.get("url") or "").strip()
        if not source_url:
            continue
        video_hash = (data.get("video_hash") or "").strip() or metadata_file.stem
        if not video_hash:
            continue
        key = _normalize_url(source_url)
        if key and key not in index:
            index[key] = video_hash
    return index


def _build_api_url_index(
    *,
    api_url: str,
    api_token: str,
    start_time: int | None = None,
    end_time: int | None = None,
) -> dict[str, str]:
    from rainrag.web_metadata_api import WebMetadataAPIClient

    index: dict[str, str] = {}
    client = WebMetadataAPIClient(base_url=api_url, token=api_token)

    if start_time is None and end_time is None:
        articles = client.export_batch()
    elif start_time is not None and end_time is not None:
        # API allows max 180-day range per request. Walk in 180-day chunks.
        articles = []
        chunk_seconds = 180 * 24 * 60 * 60
        cur = start_time
        while cur <= end_time:
            chunk_end = min(cur + chunk_seconds - 1, end_time)
            articles.extend(client.export_batch(start_time=cur, end_time=chunk_end))
            cur = chunk_end + 1
    else:
        # If only one bound is given, pass through directly.
        articles = client.export_batch(start_time=start_time, end_time=end_time)

    for article in articles:
        source_url = (article.get("url") or "").strip()
        video_hash = (article.get("video_hash") or "").strip()
        if not source_url or not video_hash:
            continue
        key = _normalize_url(source_url)
        if key and key not in index:
            index[key] = video_hash
    return index


def _extract_urls_from_cell(cell: str) -> list[str]:
    urls = re.findall(r"https?://[^\s\"'<>]+", cell or "")
    if urls:
        return urls
    v = (cell or "").strip()
    return [v] if v.startswith(("http://", "https://")) else []


def _fetch_page_title(url: str) -> str | None:
    req = Request(url, headers={"User-Agent": "rainrag-sheet-hash-lookup/1.0"})
    with urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    og = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if og and og.group(1).strip():
        return og.group(1).strip()
    title = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if title:
        text = re.sub(r"\s+", " ", title.group(1)).strip()
        if text:
            return text
    return None


def _search_hash_by_title(
    *,
    title: str,
    api_base: str,
    api_token: str | None,
    limit: int = 10,
) -> tuple[str | None, str | None]:
    q = urlencode({"q": title, "limit": limit})
    url = f"{api_base.rstrip('/')}/search-by-name?{q}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "rainrag-sheet-hash-lookup/1.0",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    results = payload.get("results") or []
    if not results:
        return None, None

    title_norm = title.strip().lower()
    exact = next((r for r in results if str(r.get("name", "")).strip().lower() == title_norm), None)
    chosen = exact or results[0]
    return chosen.get("video_hash"), chosen.get("name")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read Google Sheet links from a column/row and resolve each link to a "
            "video hash using local web_metadata JSON files."
        )
    )
    parser.add_argument("sheet_url", help="Google Spreadsheet URL")
    parser.add_argument(
        "--metadata-path",
        default="web_metadata",
        help="Path to web metadata JSON directory (default: web_metadata)",
    )
    parser.add_argument(
        "--column",
        default="H",
        help="Column letter to read links from (default: H)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Optional 1-based row number (if set, read only this row in the selected column)",
    )
    parser.add_argument(
        "--gid",
        default=None,
        help="Optional sheet tab gid override (default: auto from URL or 0)",
    )
    parser.add_argument(
        "--google-access-token",
        default=None,
        help=(
            "OAuth Bearer token for private sheets. If set (or via "
            "--google-access-token-env), use Google Sheets API instead of public CSV export."
        ),
    )
    parser.add_argument(
        "--google-access-token-env",
        default="GOOGLE_ACCESS_TOKEN",
        help="Env var name to read OAuth token from when --google-access-token is not set",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help=(
            "Optional explicit tab name for private-sheet mode; if omitted, resolved from gid"
        ),
    )
    parser.add_argument(
        "--service-account-file",
        default=None,
        help=(
            "Path to a Google service-account JSON key file. "
            "If provided, the script mints an access token automatically."
        ),
    )
    parser.add_argument(
        "--service-account-env",
        default="GOOGLE_APPLICATION_CREDENTIALS",
        help=(
            "Env var name containing path to service-account JSON key file "
            "(default: GOOGLE_APPLICATION_CREDENTIALS)"
        ),
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print JSON instead of tab-separated lines",
    )
    parser.add_argument(
        "--metadata-source",
        choices=["local", "api", "hybrid"],
        default="hybrid",
        help="Where to resolve URL->hash mapping from (default: hybrid)",
    )
    parser.add_argument(
        "--api-url",
        default="https://library.tvrain.tv",
        help="Web metadata API base URL for api/hybrid mode",
    )
    parser.add_argument(
        "--api-token-env",
        default="LIBRARY_API_TOKEN",
        help="Env var containing metadata API bearer token (default: LIBRARY_API_TOKEN)",
    )
    parser.add_argument(
        "--api-start-date",
        default=None,
        help="Optional API export start date in YYYY-MM-DD (UTC), for wider historical coverage",
    )
    parser.add_argument(
        "--api-end-date",
        default=None,
        help="Optional API export end date in YYYY-MM-DD (UTC), default is now when start date is set",
    )
    parser.add_argument(
        "--title-fallback",
        action="store_true",
        help="If URL lookup misses, fetch page title and query search-by-name API for hash",
    )
    parser.add_argument(
        "--title-search-api-base",
        default="https://rag.tvrain.tv/api",
        help="Base URL for name search API (default: https://rag.tvrain.tv/api)",
    )
    parser.add_argument(
        "--title-search-api-token-env",
        default=None,
        help="Optional env var for Bearer token to call title search API",
    )
    parser.add_argument(
        "--write-hashes-to-column",
        default=None,
        help="Optional target column letter to write resolved hashes back (e.g. G)",
    )
    parser.add_argument(
        "--write-only-empty",
        action="store_true",
        help="When writing hashes, skip rows whose target cell already has a value",
    )
    args = parser.parse_args()

    want_write = bool(args.write_hashes_to_column)
    access_token = (args.google_access_token or "").strip()
    if not access_token and args.google_access_token_env:
        access_token = os.environ.get(args.google_access_token_env, "").strip()
    if not access_token:
        service_account_file = (args.service_account_file or "").strip()
        if not service_account_file and args.service_account_env:
            service_account_file = os.environ.get(args.service_account_env, "").strip()
        if service_account_file:
            access_token = _mint_access_token_from_service_account(
                service_account_file, writable_sheets=want_write
            )

    sheet_title_for_write: str | None = None
    spreadsheet_id_for_write: str | None = None
    if access_token:
        spreadsheet_id_for_write = _parse_sheet_id(args.sheet_url)
        final_gid_for_write = args.gid if args.gid is not None else _parse_gid(args.sheet_url)
        sheet_title_for_write = (args.sheet_name or "").strip() or _resolve_sheet_title(
            spreadsheet_id_for_write, final_gid_for_write, access_token
        )
        links = _fetch_sheet_values_private(
            args.sheet_url,
            args.gid,
            access_token=access_token,
            column=args.column,
            row=args.row,
            sheet_name=sheet_title_for_write,
        )
    else:
        rows = _fetch_sheet_csv(args.sheet_url, args.gid)
        links = _iter_links(rows, args.column, args.row)
        if want_write:
            raise RuntimeError("Writing hashes requires private-sheet auth (access token/service account)")
    local_index: dict[str, str] = {}
    api_index: dict[str, str] = {}
    if args.metadata_source in {"local", "hybrid"}:
        local_index = _build_url_index(Path(args.metadata_path))
    if args.metadata_source in {"api", "hybrid"}:
        api_token = os.environ.get(args.api_token_env, "").strip()
        if not api_token:
            raise RuntimeError(
                f"metadata-source={args.metadata_source!r} requires API token in "
                f"env var {args.api_token_env!r}"
            )
        api_start_ts: int | None = None
        api_end_ts: int | None = None
        if args.api_start_date:
            start_dt = datetime.strptime(args.api_start_date, "%Y-%m-%d").replace(tzinfo=UTC)
            if args.api_end_date:
                end_dt = datetime.strptime(args.api_end_date, "%Y-%m-%d").replace(tzinfo=UTC)
                end_dt = end_dt + timedelta(days=1) - timedelta(seconds=1)
            else:
                end_dt = datetime.now(UTC)
            api_start_ts = int(start_dt.timestamp())
            api_end_ts = int(end_dt.timestamp())
            if api_start_ts > api_end_ts:
                raise RuntimeError("api-start-date must be <= api-end-date")

        api_index = _build_api_url_index(
            api_url=args.api_url,
            api_token=api_token,
            start_time=api_start_ts,
            end_time=api_end_ts,
        )

    title_api_token: str | None = None
    if args.title_search_api_token_env:
        title_api_token = os.environ.get(args.title_search_api_token_env, "").strip() or None

    results: list[dict[str, Any]] = []
    for row_num, cell_value in links:
        for url in _extract_urls_from_cell(cell_value):
            normalized = _normalize_url(url)
            video_hash = local_index.get(normalized) or api_index.get(normalized)
            matched_by = "url_index" if video_hash else None
            title_used: str | None = None
            title_match_name: str | None = None

            if not video_hash and args.title_fallback:
                try:
                    title_used = _fetch_page_title(url)
                    if title_used:
                        video_hash, title_match_name = _search_hash_by_title(
                            title=title_used,
                            api_base=args.title_search_api_base,
                            api_token=title_api_token,
                        )
                        if video_hash:
                            matched_by = "title_search"
                except Exception:
                    pass

            results.append(
                {
                    "row": row_num,
                    "column": args.column.upper(),
                    "url": url,
                    "normalized_url": normalized,
                    "video_hash": video_hash,
                    "found": video_hash is not None,
                    "matched_by": matched_by,
                    "title_used": title_used,
                    "title_match_name": title_match_name,
                }
            )

    if args.output_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("row\tcolumn\tvideo_hash\turl")
        for item in results:
            print(
                f"{item['row']}\t{item['column']}\t"
                f"{item['video_hash'] or 'NOT_FOUND'}\t{item['url']}"
            )

    if want_write:
        assert spreadsheet_id_for_write is not None
        assert sheet_title_for_write is not None
        assert access_token

        row_to_hash: dict[int, str] = {}
        for item in results:
            row_num = int(item["row"])
            vhash = item.get("video_hash")
            if not vhash:
                continue
            row_to_hash.setdefault(row_num, str(vhash))

        if args.write_only_empty and row_to_hash:
            existing = _fetch_sheet_values_private(
                args.sheet_url,
                args.gid,
                access_token=access_token,
                column=args.write_hashes_to_column,
                row=None,
                sheet_name=sheet_title_for_write,
            )
            occupied_rows = {r for r, v in existing if (v or "").strip()}
            row_to_hash = {r: h for r, h in row_to_hash.items() if r not in occupied_rows}

        updated_cells = _write_hashes_to_sheet(
            spreadsheet_id=spreadsheet_id_for_write,
            sheet_title=sheet_title_for_write,
            access_token=access_token,
            target_column=args.write_hashes_to_column,
            row_to_hash=row_to_hash,
        )
        print(
            f"Updated {updated_cells} cells in column "
            f"{args.write_hashes_to_column.upper()} (rows with resolved hash: {len(row_to_hash)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
