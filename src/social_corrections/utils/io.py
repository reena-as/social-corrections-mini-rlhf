"""Small IO helpers used across scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the absolute path to the repo root regardless of cwd."""
    # This file is at src/social_corrections/utils/io.py
    here = Path(__file__).resolve()
    return here.parents[3]


def data_dir() -> Path:
    return project_root() / "data"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Environment variable {name!r} is required. "
            f"Copy .env.example to .env and fill it in."
        )
    return value
