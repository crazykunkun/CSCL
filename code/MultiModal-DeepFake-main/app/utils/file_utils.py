from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "app" / "assets"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = resolve_path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_text(path: str | Path, default: str = "") -> str:
    path = resolve_path(path)
    if not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return default


def list_existing(paths: list[str | Path]) -> list[Path]:
    return [resolve_path(p) for p in paths if resolve_path(p).exists()]
