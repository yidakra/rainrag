"""Minimal client for the API's server-sent event streams.

Shared by the Streamlit UI (sync) and the Slack connector (async), so the two
front-ends cannot drift in how they read /query/stream. Deliberately
stdlib+httpx only: the Slack connector imports this, and it must stay free of
the query-engine dependency chain.

The server emits events as ``event: <name>\\ndata: <json>\\n\\n``; ``data`` is
always a JSON document except for no event we currently send, so the parser
decodes it eagerly and raises on garbage rather than passing it along.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

import httpx


def parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, Any]]:
    """Yield (event, decoded-data) pairs from an iterable of SSE lines.

    Pure and incremental: feed it lines as they arrive and events come out as
    soon as their terminating blank line does. Unknown fields (id:, retry:,
    comments) are ignored per the SSE spec; multi-line data fields are joined
    with newlines as the spec requires.
    """
    event: str | None = None
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n") if isinstance(raw, str) else raw.decode().rstrip("\r\n")
        if line == "":
            if event is not None and data_parts:
                yield event, json.loads("\n".join(data_parts))
            event = None
            data_parts = []
        elif line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data_parts.append(line[len("data: ") :])
        elif line.startswith("data:"):
            data_parts.append(line[len("data:") :])


def stream_query(
    api_base: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    verify: bool = True,
) -> Iterator[tuple[str, Any]]:
    """POST /query/stream and yield its (event, data) pairs synchronously.

    Raises httpx.HTTPStatusError before the first event when the endpoint
    rejects the request (401, 404 on an older API, 429), so callers can fall
    back to the blocking /query while nothing has been rendered yet.
    """
    url = f"{api_base.rstrip('/')}/query/stream"
    with httpx.Client(timeout=timeout, verify=verify) as client:  # noqa: S113 - explicit timeout
        with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            yield from parse_sse_lines(response.iter_lines())


async def astream_query(
    api_base: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
) -> AsyncIterator[tuple[str, Any]]:
    """Async twin of stream_query, for the Slack connector."""
    url = f"{api_base.rstrip('/')}/query/stream"
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()

            # parse_sse_lines is a sync generator; drive it line by line so
            # the async iteration stays incremental.
            event: str | None = None
            data_parts: list[str] = []
            async for raw in response.aiter_lines():
                line = raw.rstrip("\r\n")
                if line == "":
                    if event is not None and data_parts:
                        yield event, json.loads("\n".join(data_parts))
                    event = None
                    data_parts = []
                elif line.startswith("event: "):
                    event = line[len("event: ") :]
                elif line.startswith("data: "):
                    data_parts.append(line[len("data: ") :])
                elif line.startswith("data:"):
                    data_parts.append(line[len("data:") :])
