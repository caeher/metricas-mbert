"""Ajuste de umbrales y calibración para maximizar F1-macro (Fase 3)."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.models.mbert import predict
from discurso_odio.paths import REPORTS_METRICS

THRESHOLDS_JSON = REPORTS_METRICS / "mbert_thresholds.json"
NUM_CLASSES = load_settings()["labels"]["num_classes"]


def apply_class_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Predice clase con mayor margen sobre umbral OvR."""
    margins = probs - thresholds.reshape(1, -1)
    return np.argmax(margins, axis=1)


def grid_search_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: list[float] | None = None,
) -> tuple[np.ndarray, float]:
    """Búsqueda en rejilla de umbrales por clase (val only)."""
    grid = grid or [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_f1 = -1.0
    best_thresholds = np.full(NUM_CLASSES, 0.25, dtype=float)

    for combo in product(grid, repeat=NUM_CLASSES):
        thresholds = np.array(combo, dtype=float)
        preds = apply_class_thresholds(y_prob, thresholds)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thresholds = thresholds.copy()

    return best_thresholds, best_f1


def tune_thresholds_on_val(
    model_dir: str | Path,
    *,
    grid: list[float] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Ajusta umbrales en val y exporta JSON."""
    val_df = load_split("val")
    texts = val_df["texto_modelo"].fillna("").astype(str).tolist()
    y_true = val_df["label"].astype(int).to_numpy()
    _, probs = predict(texts, model_dir=model_dir)

    thresholds, val_f1 = grid_search_thresholds(y_true, probs, grid=grid)
    preds = apply_class_thresholds(probs, thresholds)
    baseline_f1 = float(f1_score(y_true, np.argmax(probs, axis=1), average="macro", zero_division=0))

    payload = {
        "model_dir": str(model_dir),
        "thresholds": thresholds.tolist(),
        "val_f1_macro_baseline_argmax": baseline_f1,
        "val_f1_macro_tuned": val_f1,
        "val_gain_pp": float(val_f1 - baseline_f1),
    }
    out = output_path or THRESHOLDS_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_thresholds(path: Path | None = None) -> np.ndarray | None:
    path = path or THRESHOLDS_JSON
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["thresholds"], dtype=float)
