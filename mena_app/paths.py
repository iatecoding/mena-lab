"""Where things live. Nothing here may contain a machine-specific path: models come from models.py and the user's config."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
EXAMPLES_DIR = APP_DIR / "examples"
DATA_DIR = APP_DIR / "data"


def data_dir() -> Path:
    """Per-user folder for job files, the model list and this machine's calibration (outside the repo)."""
    d = Path(os.environ.get("MENA_HOME", str(Path.home() / ".mena"))).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d
