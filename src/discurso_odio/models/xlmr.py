"""Inferencia batch con checkpoint XLM-RoBERTa."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from discurso_odio.models.inference_contract import (
    InferenceContract,
    apply_normalization,
    get_inference_contract,
    resolve_inference_contract,
    write_inference_contract,
)
from discurso_odio.paths import PROJECT_ROOT


def default_xlmr_model_dir() -> Path:
    return PROJECT_ROOT / "XLMR" / "02_modelo_prototipo" / "modelo_entrenado"


def make_tokenize_function(tokenizer, contract: InferenceContract):
    """Devuelve una función de tokenización compatible con Hugging Face datasets."""

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        col = contract.text_column
        return tokenizer(
            examples[col],
            truncation=True,
            max_length=contract.max_length,
        )

    return tokenize_function


def predict(
    texts: list[str],
    model_dir: Path | str | None = None,
    *,
    max_length: int | None = None,
    batch_size: int = 16,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Predice etiquetas y probabilidades para una lista de textos."""
    model_dir = Path(model_dir) if model_dir else default_xlmr_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint XLM-R no encontrado: {model_dir}")

    contract = resolve_inference_contract(model_dir)
    max_length = contract.validate_max_length(max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalize:
        texts = apply_normalization(texts, contract)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=-1))

    return np.concatenate(all_preds), np.concatenate(all_probs)


def ensure_checkpoint_contract(model_dir: Path | str | None = None) -> Path:
    """Escribe inference_contract.json en el directorio del checkpoint."""
    model_dir = Path(model_dir) if model_dir else default_xlmr_model_dir()
    return write_inference_contract(model_dir, get_inference_contract())
