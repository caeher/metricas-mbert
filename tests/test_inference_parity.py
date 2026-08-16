"""Pruebas de paridad entrenamiento / evaluación / inferencia XLM-R (T-06)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.features.normalization import normalize_for_model
from discurso_odio.models.inference_contract import (
    apply_normalization_single,
    get_inference_contract,
    resolve_inference_contract,
)
from discurso_odio.models.xlmr import default_xlmr_model_dir, predict
from discurso_odio.paths import PROJECT_ROOT, TEST_CSV

XLMR_ROOT = PROJECT_ROOT / "XLMR"
MODEL_DIR = default_xlmr_model_dir()
INFERENCIA_PATH = XLMR_ROOT / "02_modelo_prototipo" / "inferencia.py"

PARITY_SCAN_PATHS = [
    PROJECT_ROOT / "src" / "discurso_odio" / "models" / "xlmr.py",
    PROJECT_ROOT / "scripts" / "export_xlmr_predictions.py",
    INFERENCIA_PATH,
    XLMR_ROOT / "02_modelo_prototipo" / "estadisticas.ipynb",
]

SYNTHETIC_TEXTS = [
    "Hola @usuario123 mira https://t.co/abc",
    "Ese maje está loco 😂",
    "Te voy a romper la cara",
]


def _load_inferencia_module():
    spec = importlib.util.spec_from_file_location("xlmr_inferencia", INFERENCIA_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has_model_weights() -> bool:
    return MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors").exists()


@pytest.mark.parametrize("path", PARITY_SCAN_PATHS)
def test_no_max_length_256_in_xlmr_paths(path: Path):
    assert path.exists(), f"Archivo no encontrado: {path}"
    content = path.read_text(encoding="utf-8")
    assert not re.search(r"max_length\s*=\s*256", content), f"max_length=256 en {path}"
    assert not re.search(r"max_length:\s*256", content), f"max_length: 256 en {path}"


def test_inference_contract_settings():
    contract = get_inference_contract()
    assert contract.max_length == 128
    assert contract.normalizer == "normalize_for_model"
    assert contract.normalizer_version == "1.1"
    assert contract.text_column == "texto_modelo"


def test_normalization_idempotent_on_texto_modelo():
    test = load_split("test").head(50)
    for texto in test["texto_modelo"]:
        once = normalize_for_model(str(texto))
        twice = normalize_for_model(once)
        assert once == twice


def test_normalization_matches_pipeline_on_sample():
    test = load_split("test").head(30)
    mismatches = 0
    for _, row in test.iterrows():
        expected = str(row["texto_modelo"])
        computed = normalize_for_model(str(row["texto_original"]))
        if computed != expected:
            mismatches += 1
    assert mismatches / len(test) < 0.05, (
        f"Demasiadas discrepancias normalización vs texto_modelo: {mismatches}/{len(test)}"
    )


def test_synthetic_normalization_endpoint_parity():
    contract = get_inference_contract()
    for text in SYNTHETIC_TEXTS:
        assert apply_normalization_single(text, contract) == normalize_for_model(text)


@pytest.mark.skipif(not _has_model_weights(), reason="Checkpoint XLM-R no disponible")
def test_token_ids_eval_vs_endpoint():
    from transformers import AutoTokenizer

    contract = resolve_inference_contract(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    test = load_split("test").head(50)

    aligned = test[
        test.apply(
            lambda row: normalize_for_model(str(row["texto_original"])) == str(row["texto_modelo"]),
            axis=1,
        )
    ].head(20)
    assert len(aligned) >= 5, "Muestra alineada insuficiente para prueba de tokens"

    for _, row in aligned.iterrows():
        eval_text = apply_normalization_single(str(row["texto_modelo"]), contract)
        endpoint_text = apply_normalization_single(str(row["texto_original"]), contract)
        eval_ids = tokenizer(eval_text, truncation=True, max_length=contract.max_length)["input_ids"]
        endpoint_ids = tokenizer(
            endpoint_text, truncation=True, max_length=contract.max_length
        )["input_ids"]
        assert eval_text == endpoint_text
        assert eval_ids == endpoint_ids


@pytest.mark.skipif(not _has_model_weights(), reason="Checkpoint XLM-R no disponible")
def test_predictions_batch_vs_endpoint():
    inferencia = _load_inferencia_module()
    tokenizer, model, device, contract = inferencia.cargar_modelo()
    test = load_split("test").head(32)

    batch_preds, _ = predict(
        test["texto_original"].tolist(),
        MODEL_DIR,
        normalize=True,
    )

    endpoint_preds = []
    for text in test["texto_original"]:
        clase, _, _ = inferencia.predecir_toxicidad(text, tokenizer, model, device, contract)
        idx = inferencia.LABELS.index(clase)
        endpoint_preds.append(idx)

    np.testing.assert_array_equal(batch_preds, np.asarray(endpoint_preds))


@pytest.mark.skipif(not _has_model_weights(), reason="Checkpoint XLM-R no disponible")
def test_max_length_override_rejected():
    with pytest.raises(ValueError, match="no coincide con el contrato"):
        predict(["hola"], MODEL_DIR, max_length=256)
