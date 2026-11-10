"""loguru formats with str.format, so printf placeholders are printed literally.

A `logger.warning("dropped %d/%d", a, b)` call under loguru logs the format
string as written and silently discards both numbers (#45). This walks every
module that imports loguru's logger and fails on any call that passes
arguments with a printf-style message and no brace field.
"""

import ast
import re
from pathlib import Path

import pytest
from loguru import logger


SRC = Path(__file__).resolve().parents[2] / "src"
PRINTF = re.compile(r"%[-+ #0]*\d*(?:\.\d+)?[sdifrx]")
LOG_METHODS = {
    "trace",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "exception",
    "log",
}


def _uses_loguru(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "loguru"
        and any(alias.name == "logger" for alias in node.names)
        for node in tree.body
    )


def _literal(node: ast.expr) -> str | None:
    """The message text, including implicit and `+` concatenation of literals."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _printf_calls() -> list[str]:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _uses_loguru(tree):
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and node.func.attr in LOG_METHODS
            ):
                continue
            args = node.args[1:] if node.func.attr == "log" else node.args
            if len(args) < 2:
                continue
            message = _literal(args[0])
            if message and PRINTF.search(message) and "{" not in message:
                found.append(f"{path.relative_to(SRC)}:{node.lineno}: {message!r}")
    return found


def test_loguru_calls_use_brace_placeholders():
    assert _printf_calls() == []


def test_the_premise_loguru_drops_printf_arguments():
    """Guard the reason for the test above: if loguru ever learned printf, it
    could go."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), format="{message}")
    try:
        logger.warning("dropped %d/%d", 3, 20)
        logger.warning("dropped {}/{}", 3, 20)
    finally:
        logger.remove(sink)
    assert messages == ["dropped %d/%d", "dropped 3/20"]


@pytest.mark.parametrize("bad", ["x=%d", "x=%s", "x=%.3f", "x=%r"])
def test_the_detector_sees_printf(bad):
    assert PRINTF.search(bad)
