"""Ensamble de checkpoints mBERT por promedio de probabilidades (Fase 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.thresholds import apply_class_thresholds, load_thresholds
from discurso_odio.models.mbert import predict
from discurso_odio.paths import REPORTS_METRICS

ENSEMBLE_JSON = REPORTS_METRICS / "mbert_ensemble_manifest.json"


def ensemble_predict(
    texts: list[str],
    model_dirs: list[str | Path],
    *,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Promedia probabilidades de varios checkpoints mBERT."""
    if not model_dirs:
        raise ValueError("Se requiere al menos un checkpoint")

    weights = weights or [1.0 / len(model_dirs)] * len(model_dirs)
    if len(weights) != len(model_dirs):
        raise ValueError("weights debe tener la misma longitud que model_dirs")

    total = None
    for model_dir, w in zip(model_dirs, weights):
        _, probs = predict(texts, model_dir=model_dir)
        weighted = probs * float(w)
        total = weighted if total is None else total + weighted

    assert total is not None
    preds = np.argmax(total, axis=1)
    return preds, total


def evaluate_ensemble_on_split(
    model_dirs: list[str | Path],
    split: str = "val",
    *,
    weights: list[float] | None = None,
    thresholds_path: Path | None = None,
) -> dict[str, Any]:
    """Evalúa ensamble en un split."""
    df = load_split(split)
    texts = df["texto_modelo"].fillna("").astype(str).tolist()
    y_true = df["label"].astype(int).to_numpy()

    preds, probs = ensemble_predict(texts, model_dirs, weights=weights)
    f1_argmax = float(f1_score(y_true, preds, average="macro", zero_division=0))

    thresholds = load_thresholds(thresholds_path)
    f1_tuned = f1_argmax
    if thresholds is not None:
        tuned_preds = apply_class_thresholds(probs, thresholds)
        f1_tuned = float(f1_score(y_true, tuned_preds, average="macro", zero_division=0))

    return {
        "split": split,
        "n": int(len(df)),
        "model_dirs": [str(p) for p in model_dirs],
        "f1_macro_argmax": f1_argmax,
        "f1_macro_with_thresholds": f1_tuned,
    }


def export_ensemble_manifest(
    model_dirs: list[str | Path],
    *,
    weights: list[float] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evalúa ensamble en val y exporta manifiesto."""
    val_metrics = evaluate_ensemble_on_split(model_dirs, "val", weights=weights)
    payload = {
        "model_dirs": [str(p) for p in model_dirs],
        "weights": weights or [1.0 / len(model_dirs)] * len(model_dirs),
        "val_metrics": val_metrics,
    }
    out = output_path or ENSEMBLE_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
