"""A small shim package providing a safer import name for the evaluation tools.

The upstream codebase historically used the top-level package name ``eval``.
That name shadows Python's built-in ``eval()`` function and can be confusing
in type-checkers and linters.

This package re-exports the public API of ``eval`` so code can write:

    import rainrag_eval as eval

or simply:

    import rainrag_eval

without changing the underlying implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

# Re-export everything from the old `eval` package for backward compatibility.
# This shim is intentionally minimal; it is primarily to provide a less
# confusing import name in user code.
import eval as _eval
from eval import *  # noqa: F401,F403


eval_all: Sequence[str] | None = getattr(_eval, "__all__", None)
if eval_all is not None:
    __all__ = list(eval_all)  # pyright: ignore[reportUnsupportedDunderAll]
else:
    __all__ = [name for name in dir(_eval) if not name.startswith("_")]  # pyright: ignore[reportUnsupportedDunderAll]
