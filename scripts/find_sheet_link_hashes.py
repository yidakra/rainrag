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


def _mint_access_token_from_service_account(service_account_file: str) -> str:
    """Mint an OAuth access token from a service-account JSON key file."""
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "google-auth is required for service-account mode. "
            "Install it with: uv add google-auth"
        ) from exc

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
    args = parser.parse_args()

    access_token = (args.google_access_token or "").strip()
    if not access_token and args.google_access_token_env:
        access_token = os.environ.get(args.google_access_token_env, "").strip()
    if not access_token:
        service_account_file = (args.service_account_file or "").strip()
        if not service_account_file and args.service_account_env:
            service_account_file = os.environ.get(args.service_account_env, "").strip()
        if service_account_file:
            access_token = _mint_access_token_from_service_account(service_account_file)

    if access_token:
        links = _fetch_sheet_values_private(
            args.sheet_url,
            args.gid,
            access_token=access_token,
            column=args.column,
            row=args.row,
            sheet_name=args.sheet_name,
        )
    else:
        rows = _fetch_sheet_csv(args.sheet_url, args.gid)
        links = _iter_links(rows, args.column, args.row)
    url_index = _build_url_index(Path(args.metadata_path))

    results: list[dict[str, Any]] = []
    for row_num, url in links:
        normalized = _normalize_url(url)
        video_hash = url_index.get(normalized)
        results.append(
            {
                "row": row_num,
                "column": args.column.upper(),
                "url": url,
                "normalized_url": normalized,
                "video_hash": video_hash,
                "found": video_hash is not None,
            }
        )

    if args.output_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print("row\tcolumn\tvideo_hash\turl")
    for item in results:
        print(
            f"{item['row']}\t{item['column']}\t"
            f"{item['video_hash'] or 'NOT_FOUND'}\t{item['url']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
