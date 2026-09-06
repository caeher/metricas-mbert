"""Explicaciones LIME para mBERT."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

import numpy as np
from lime.lime_text import LimeTextExplainer

from discurso_odio.config import load_settings

_WORDPIECE_SUFFIX = re.compile(r"^##")


def normalize_tokens_for_jaccard(tokens: list[str]) -> list[str]:
    """Fusiona subwords WordPiece (##) y normaliza a palabras comparables con LIME."""
    merged: list[str] = []
    for token in tokens:
        piece = str(token).strip()
        if not piece:
            continue
        if _WORDPIECE_SUFFIX.match(piece):
            suffix = piece[2:]
            if merged:
                merged[-1] = merged[-1] + suffix
            elif suffix:
                merged.append(suffix)
        else:
            merged.append(piece)
    cleaned = [t.lower().strip(".,!?;:\"'()[]") for t in merged if t.strip()]
    return [t for t in cleaned if t and t not in {"<pad>", "<s>", "</s>", "[PAD]", "[cls]", "[sep]"}]


def build_lime_explainer(class_names: list[str] | None = None) -> LimeTextExplainer:
    settings = load_settings()
    class_names = class_names or settings["labels"]["names"]
    return LimeTextExplainer(class_names=class_names, random_state=settings["project"]["seed"])


def make_predict_proba_fn(predict_fn: Callable[[list[str]], np.ndarray]) -> Callable:
    """Crea callable para LIME que devuelve probabilidades (n_samples, n_classes)."""

    def _predict(texts: list[str]) -> np.ndarray:
        texts = [str(t) if t else " " for t in texts]
        probs = predict_fn(texts)
        if isinstance(probs, tuple):
            _, probs = probs
        return np.asarray(probs, dtype=np.float64)

    return _predict


def explain_instance(
    text: str,
    predict_proba_fn: Callable,
    explainer: LimeTextExplainer | None = None,
    num_features: int = 10,
    num_samples: int = 500,
) -> dict:
    if not str(text).strip():
        return {"text": text, "predicted_class": -1, "weights": [], "top_tokens": []}

    explainer = explainer or build_lime_explainer()
    exp = explainer.explain_instance(
        text,
        predict_proba_fn,
        num_features=num_features,
        num_samples=num_samples,
    )
    if not exp.top_labels:
        return {"text": text, "predicted_class": -1, "weights": [], "top_tokens": []}

    pred_class = exp.top_labels[0]
    weights = exp.as_list(label=pred_class)
    return {
        "text": text,
        "predicted_class": pred_class,
        "weights": weights,
        "top_tokens": [w[0] for w in weights[:num_features]],
    }


def aggregate_lime_weights(explanations: list[dict], top_k: int = 10) -> dict[int, list[tuple[str, float]]]:
    """Agrega importancia global de tokens por clase."""
    agg: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for exp in explanations:
        cls = exp["predicted_class"]
        if cls < 0:
            continue
        for token, weight in exp["weights"]:
            agg[cls][token] += abs(weight)

    result = {}
    for cls, tokens in agg.items():
        sorted_tokens = sorted(tokens.items(), key=lambda x: -x[1])[:top_k]
        result[cls] = sorted_tokens
    return result


def jaccard_top_k(tokens_a: list[str], tokens_b: list[str], k: int = 10) -> float:
    norm_a = normalize_tokens_for_jaccard(tokens_a)
    norm_b = normalize_tokens_for_jaccard(tokens_b)
    set_a = set(norm_a[:k])
    set_b = set(norm_b[:k])
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
