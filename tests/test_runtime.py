"""Tests del predictor en memoria."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from discurso_odio.models.inference_contract import InferenceContract
from discurso_odio.models.runtime import CachedPredictor, resolve_model_dir


def test_resolve_model_dir_mbert_default():
    path = resolve_model_dir(backend="mbert")
    assert path.name == "mbert-sv"


def test_resolve_model_dir_xlmr():
    path = resolve_model_dir(backend="xlmr")
    assert "modelo_entrenado" in str(path)


@patch("discurso_odio.models.runtime.AutoModelForSequenceClassification.from_pretrained")
@patch("discurso_odio.models.runtime.AutoTokenizer.from_pretrained")
def test_cached_predictor_normalizes(mock_tokenizer_cls, mock_model_cls, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "inference_contract.json").write_text(
        '{"max_length": 128, "text_column": "texto_modelo", '
        '"normalizer": "normalize_for_model", "normalizer_version": "1.1"}',
        encoding="utf-8",
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.return_value = mock_tokenizer

    import torch

    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_model.eval.return_value = mock_model

    logits = MagicMock()
    logits.logits = torch.tensor([[1.0, 2.0, 0.5, 0.1]])
    mock_model.return_value = logits

    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    predictor = CachedPredictor(model_dir=model_dir)
    idx, conf, probs = predictor.predict_single("hola mundo")

    assert idx == 1
    assert conf > 0
    assert len(probs) == 4
    mock_tokenizer.assert_called()
