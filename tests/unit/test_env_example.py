"""Every env var that a config field names must be listed in .env.example.

A field such as ``notify.token_env`` or ``web_metadata.api_token_env`` holds the
name of an environment variable. A deploy built from .env.example never sets a
variable the file does not list, so the feature behind it silently runs without
its secret (#128).
"""

import inspect
import re
from pathlib import Path

from pydantic import BaseModel

from rainrag import config as config_module


ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def _env_name_defaults() -> dict[str, str]:
    """Return {"Class.field": default} for every str field whose name ends in _env."""
    found: dict[str, str] = {}
    for name, cls in inspect.getmembers(config_module, inspect.isclass):
        if not issubclass(cls, BaseModel) or cls.__module__ != config_module.__name__:
            continue
        for field_name, field in cls.model_fields.items():
            if field_name.endswith("_env") and isinstance(field.default, str) and field.default:
                found[f"{name}.{field_name}"] = field.default
    return found


def test_env_name_fields_are_found() -> None:
    """The scan sees the notify webhook token field, so it is not vacuous."""
    defaults = _env_name_defaults()
    assert defaults.get("NotifyConfig.token_env") == "RAINRAG_NOTIFY_TOKEN"


def test_every_env_name_default_is_listed_in_env_example() -> None:
    """Each default env var name appears as VAR= or # VAR= in .env.example."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    missing = {
        field: var
        for field, var in _env_name_defaults().items()
        if not re.search(rf"^#?\s*{re.escape(var)}=", text, flags=re.MULTILINE)
    }
    assert not missing, f".env.example does not list: {missing}"
