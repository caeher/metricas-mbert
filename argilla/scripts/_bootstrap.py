"""Shared path setup for Argilla CLI scripts under argilla/scripts/."""

from __future__ import annotations

import sys
from pathlib import Path

ARGILLA_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARGILLA_SCRIPTS_DIR.parents[1]


def configure_argilla_script_path() -> Path:
    """Insert argilla/scripts and src on sys.path; return repo root."""
    scripts = str(ARGILLA_SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return REPO_ROOT
