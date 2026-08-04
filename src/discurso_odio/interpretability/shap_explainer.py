"""Explicaciones SHAP para mBERT."""

from __future__ import annotations

from typing import Callable

import numpy as np
import shap

from discurso_odio.config import load_settings


def _normalize_text_batch(texts) -> list[str]:
    """Convierte entradas SHAP (str, list[str] o tokens enmascarados) a strings."""
    if texts is None:
        return [""]
    if isinstance(texts, str):
        return [texts]
    if len(texts) == 0:
        return [""]
    first = texts[0]
    if isinstance(first, (list, tuple)):
        return [" ".join(str(t) for t in row if t is not None) for row in texts]
    return [str(t) for t in texts]


def make_shap_predict_fn(
    model,
    tokenizer,
    device,
    max_length: int | None = None,
) -> Callable:
    settings = load_settings()
    max_length = max_length or settings["model"]["max_length"]
    import torch

    def _predict(texts) -> np.ndarray:
        batch = _normalize_text_batch(texts)
        model.eval()
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            return torch.softmax(logits, dim=-1).cpu().numpy()

    return _predict


def explain_texts_shap(
    texts: list[str],
    predict_fn: Callable[[list[str]], np.ndarray],
    tokenizer=None,
    max_evals: int = 100,
) -> list:
    """SHAP sobre textos usando Text masker con tokenizador del modelo."""
    texts = [str(t) for t in texts if str(t).strip()]
    if not texts:
        return []
    if tokenizer is not None:
        masker = shap.maskers.Text(tokenizer)
    else:
        masker = shap.maskers.Text()
    explainer = shap.Explainer(predict_fn, masker)
    return explainer(texts, max_evals=max_evals)


def extract_top_tokens(shap_values, text_idx: int = 0, class_idx: int | None = None, top_k: int = 10) -> list[tuple[str, float]]:
    """Extrae top-k tokens por magnitud SHAP."""
    sv = shap_values[text_idx]
    if class_idx is None:
        values = np.abs(sv.values).sum(axis=-1) if sv.values.ndim > 1 else np.abs(sv.values)
    else:
        values = np.abs(sv.values[:, class_idx]) if sv.values.ndim > 1 else np.abs(sv.values)

    tokens = sv.data if hasattr(sv, "data") else []
    if isinstance(tokens, str):
        tokens = tokens.split()

    pairs = list(zip(tokens, values)) if len(tokens) == len(values) else []
    pairs.sort(key=lambda x: -x[1])
    return pairs[:top_k]


def dialect_attribution_mass(
    top_tokens: list[tuple[str, float]],
    jerga_tokens: set[str],
) -> float:
    """Fracción de masa SHAP/LIME en tokens de jerga."""
    if not top_tokens:
        return 0.0
    total = sum(abs(w) for _, w in top_tokens)
    jerga_mass = sum(abs(w) for t, w in top_tokens if t.lower() in jerga_tokens)
    return jerga_mass / total if total > 0 else 0.0
