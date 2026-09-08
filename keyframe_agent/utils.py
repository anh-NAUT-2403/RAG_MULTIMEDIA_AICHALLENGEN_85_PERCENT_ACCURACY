from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def slugify(text: str, limit: int = 54) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:limit] or "query"


def make_run_dir(output_root: Path, query: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{stamp}_{slugify(query)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

