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

# Re-export everything from the old `eval` package for backward compatibility.
# This shim is intentionally minimal; it is primarily to provide a less
# confusing import name in user code.
import eval as _eval
from eval import *  # noqa: F401,F403


if hasattr(_eval, "__all__"):
    __all__ = list(_eval.__all__)
else:
    __all__ = [name for name in dir(_eval) if not name.startswith("_")]
