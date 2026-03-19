"""Shim module for legacy eval.run_eval to rainrag_eval.run_eval."""

from eval.run_eval import app  # noqa: F401


__all__ = ["app"]
