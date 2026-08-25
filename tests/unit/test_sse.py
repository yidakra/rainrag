"""Tests for the shared SSE client parser.

Both front-ends read /query/stream through this parser; what matters is that
it is incremental (events come out as their blank line arrives, not at EOF)
and spec-tolerant (multi-line data, comments, unknown fields).
"""

from __future__ import annotations

import pytest

from rainrag.sse import parse_sse_lines


def test_basic_event_sequence():
    lines = [
        "event: context",
        'data: {"answer": ""}',
        "",
        "event: delta",
        'data: {"text": "от"}',
        "",
        "event: done",
        'data: {"answer": "ответ"}',
        "",
    ]
    events = list(parse_sse_lines(lines))
    assert events == [
        ("context", {"answer": ""}),
        ("delta", {"text": "от"}),
        ("done", {"answer": "ответ"}),
    ]


def test_incremental_not_eof_bound():
    """An event must be yielded as soon as its blank line arrives."""

    def lines():
        yield "event: delta"
        yield 'data: {"text": "x"}'
        yield ""
        raise AssertionError("parser consumed past the first event")

    gen = parse_sse_lines(lines())
    assert next(gen) == ("delta", {"text": "x"})


def test_multiline_data_joined_with_newlines():
    lines = ["event: delta", 'data: {"text":', 'data: "x"}', ""]
    assert list(parse_sse_lines(lines)) == [("delta", {"text": "x"})]


def test_comments_and_unknown_fields_ignored():
    lines = [": keep-alive", "id: 7", "retry: 100", "event: delta", 'data: {"text": "a"}', ""]
    assert list(parse_sse_lines(lines)) == [("delta", {"text": "a"})]


def test_blank_line_without_event_is_noise():
    assert list(parse_sse_lines(["", "", "event: done", "data: {}", ""])) == [("done", {})]


def test_bytes_lines_are_decoded():
    lines = [b"event: delta", b'data: {"text": "\xd0\xb0"}', b""]
    assert list(parse_sse_lines(lines)) == [("delta", {"text": "а"})]


def test_garbage_json_raises_rather_than_passing_through():
    with pytest.raises(ValueError):
        list(parse_sse_lines(["event: delta", "data: not json", ""]))
