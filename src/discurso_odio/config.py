"""Carga de configuración YAML y variables de entorno."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from discurso_odio.paths import CONFIG_PATH, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_run_profile() -> str:
  profile = os.getenv("RUN_PROFILE", "cpu").lower()
  return profile if profile in ("cpu", "gpu") else "cpu"


def get_training_config() -> dict[str, Any]:
    settings = load_settings()
    profile = get_run_profile()
    return settings["training"][profile]


def get_max_steps() -> int:
    return int(os.getenv("MAX_STEPS", "0"))


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or None
