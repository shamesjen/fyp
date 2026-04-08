from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class MissingCredentialError(RuntimeError):
    """Raised when required provider credentials are not available."""


def load_environment(env_path: str | Path | None = None) -> Path | None:
    """Load environment variables from the repo-local .env if it exists."""

    candidate = Path(env_path) if env_path else Path.cwd() / ".env"
    if candidate.exists():
        load_dotenv(candidate, override=False)
        return candidate
    load_dotenv(override=False)
    return None


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    load_environment()
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir else Path.cwd()
    return (base / path).resolve()


def get_alpaca_credentials(required: bool = True) -> dict[str, str | None]:
    load_environment()
    key = os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET")
    if required and (not key or not secret):
        raise MissingCredentialError(
            "Missing ALPACA_KEY or ALPACA_SECRET. Create a .env file in the repo root "
            "from .env.example and keep the variable names unchanged."
        )
    return {"key": key, "secret": secret}


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
