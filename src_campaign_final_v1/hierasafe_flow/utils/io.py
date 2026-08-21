from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_yaml(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")

