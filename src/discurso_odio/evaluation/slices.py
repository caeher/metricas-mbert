"""Métricas por cortes (plataforma, jerga, sarcasmo, ironía)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from discurso_odio.evaluation.metrics import compute_all_metrics


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["SI", "SÍ", "YES", "TRUE", "1"])


def slice_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Calcula métricas por subconjuntos relevantes."""
    results: dict[str, dict[str, Any]] = {}
    work = df.copy()
    work["_y_true"] = y_true
    work["_y_pred"] = y_pred

    if "Plataforma" in work.columns:
        for platform in work["Plataforma"].unique():
            mask = work["Plataforma"] == platform
            idx = mask.values
            prob_slice = y_prob[idx] if y_prob is not None else None
            results[f"plataforma_{platform}"] = compute_all_metrics(
                y_true[idx], y_pred[idx], prob_slice
            )

    if "n_jerga" in work.columns:
        for label, cond in [("con_jerga", work["n_jerga"] > 0), ("sin_jerga", work["n_jerga"] == 0)]:
            idx = cond.values
            prob_slice = y_prob[idx] if y_prob is not None else None
            results[label] = compute_all_metrics(y_true[idx], y_pred[idx], prob_slice)

    for col, name in [
        ("Presencia_Sarcasmo", "sarcasmo"),
        ("Presencia_Ironia", "ironia"),
        ("Contexto_Politico", "contexto_politico"),
    ]:
        if col in work.columns:
            for val, flag in [("si", True), ("no", False)]:
                mask = _to_bool(work[col]) == flag
                idx = mask.values
                if idx.sum() == 0:
                    continue
                prob_slice = y_prob[idx] if y_prob is not None else None
                results[f"{name}_{val}"] = compute_all_metrics(
                    y_true[idx], y_pred[idx], prob_slice
                )

    return results


def dialect_bias_probe(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Tasa de FP en No Tóxico con y sin jerga local."""
    if "n_jerga" not in df.columns:
        return {}

    not_toxic = y_true == 0
    fp = y_pred >= 1

    with_jerga = (df["n_jerga"] > 0).values & not_toxic
    without_jerga = (df["n_jerga"] == 0).values & not_toxic

    def _fpr(mask: np.ndarray) -> float:
        n = mask.sum()
        return float((fp & mask).sum() / n) if n > 0 else 0.0

    return {
        "fpr_no_toxico_con_jerga": _fpr(with_jerga),
        "fpr_no_toxico_sin_jerga": _fpr(without_jerga),
        "n_no_toxico_con_jerga": int(with_jerga.sum()),
        "n_no_toxico_sin_jerga": int(without_jerga.sum()),
    }
