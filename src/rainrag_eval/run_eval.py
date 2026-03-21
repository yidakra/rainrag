"""Compatibility shim providing rainrag_eval.run_eval.app as an alias for eval.run_eval.app.

This module re-exports the symbol `app` from `eval.run_eval` so that
imports of `rainrag_eval.run_eval.app` point to the implementation in
`eval.run_eval`.
"""

from eval.run_eval import app  # noqa: F401


__all__ = ["app"]
