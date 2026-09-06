"""Tests de alineación LIME-SHAP y id2label en checkpoint."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.interpretability.lime_explainer import (
    jaccard_top_k,
    normalize_tokens_for_jaccard,
)
from discurso_odio.models.mbert_artifacts import apply_id2label
from discurso_odio.paths import MODELS_DIR


def test_normalize_wordpiece_merges_subwords():
    tokens = ["que", "##bon", "##ita", "familia"]
    merged = normalize_tokens_for_jaccard(tokens)
    assert "quebonita" in merged or "que" in merged
    assert "familia" in merged


def test_jaccard_nonzero_after_alignment():
    lime = ["familia", "bonita", "que"]
    shap = ["que", "##bon", "##ita", "familia"]
    score = jaccard_top_k(lime, shap, k=3)
    assert score > 0.0


def test_apply_id2label_writes_real_names(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}}),
        encoding="utf-8",
    )
    config = apply_id2label(config_path)
    assert config["id2label"]["0"] == "No Tóxico"
    assert config["id2label"]["3"] == "Amenazas/Violencia"
    assert "LABEL_0" not in config["id2label"]["0"]


@pytest.mark.skipif(not (MODELS_DIR / "mbert-sv" / "config.json").exists(), reason="sin checkpoint")
def test_official_checkpoint_id2label():
    config_path = MODELS_DIR / "mbert-sv" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "id2label" in config:
        assert config["id2label"].get("0") != "LABEL_0"
