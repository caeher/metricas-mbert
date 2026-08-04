"""Tests de normalización."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.features.normalization import (
    normalize_aggressive,
    normalize_for_model,
    replace_mentions,
    replace_urls,
)


def test_preserve_accents():
    text = "Niño con café"
    result = normalize_for_model(text)
    assert "é" in result or "café" in result


def test_mention_replacement():
    text = "@usuario123 hola mundo"
    result = normalize_for_model(text)
    assert "@usuario" in result
    assert "@usuario123" not in result


def test_url_replacement():
    text = "mira https://t.co/abc esto"
    result = normalize_for_model(text)
    assert "http://url" in result
    assert "https://" not in result


def test_aggressive_strips_accents():
    text = "Niño con café"
    result = normalize_aggressive(text)
    assert "ñ" not in result.lower() or "nino" in result


def test_empty_text():
    assert normalize_for_model("") == ""
    assert normalize_for_model(None) == ""  # type: ignore
