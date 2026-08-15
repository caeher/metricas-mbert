"""Métricas de evaluación completas."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from discurso_odio.config import load_settings


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def ordinal_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.astype(float) - y_pred.astype(float)) ** 2)))


def ordinal_large_errors(y_true: np.ndarray, y_pred: np.ndarray, threshold: int = 2) -> float:
    """Porcentaje de errores que cruzan >= threshold niveles."""
    errors = np.abs(y_true - y_pred)
    return float(np.mean(errors >= threshold) * 100)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE para clasificación multiclase (confianza máxima)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y_true) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if not scores:
        return 0.0, 0.0, 0.0
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), lo, hi


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Prueba de McNemar entre dos clasificadores."""
    from scipy import stats

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}
    result = stats.binomtest(b, b + c, 0.5)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "b": b, "c": c}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    label_names = label_names or settings["labels"]["names"]
    n_classes = settings["labels"]["num_classes"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "ordinal_rmse": ordinal_rmse(y_true, y_pred),
        "large_error_pct": ordinal_large_errors(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "kappa_linear": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }

    for i, name in enumerate(label_names):
        metrics[f"precision_{name}"] = float(precision[i])
        metrics[f"recall_{name}"] = float(recall[i])
        metrics[f"f1_{name}"] = float(f1[i])
        metrics[f"support_{name}"] = int(support[i])

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
            )
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=list(range(n_classes))))
            metrics["ece"] = expected_calibration_error(y_true, y_prob)
            brier_scores = [
                brier_score_loss((y_true == i).astype(int), y_prob[:, i])
                for i in range(n_classes)
            ]
            metrics["brier_macro"] = float(np.mean(brier_scores))
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None
            metrics["log_loss"] = None
            metrics["ece"] = None
            metrics["brier_macro"] = None

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        target_names=label_names,
        zero_division=0,
    )
    return metrics


def trivial_baselines(class_counts: dict[int, int]) -> dict[str, float]:
    """Pisos de referencia sin modelo (informe v1.1)."""
    total = sum(class_counts.values())
    k = len(class_counts)
    props = {c: n / total for c, n in class_counts.items()}
    majority_class = max(class_counts, key=class_counts.get)

    f1_stratified = 1.0 / k
    f1_uniform = sum(2 * (1 / k) * (1 / k) / (2 / k) for _ in range(k)) / k
    f1_majority = sum(
        2 * (props[majority_class] if c == majority_class else 0.0)
        * (props[majority_class] if c == majority_class else 0.0)
        / max(
            (props[majority_class] if c == majority_class else 0.0)
            + (props[majority_class] if c == majority_class else 0.0),
            1e-9,
        )
        for c in class_counts
    ) / k

    return {
        "azar_estratificado": round(f1_stratified, 3),
        "azar_uniforme": round(f1_uniform, 3),
        "clase_mayoritaria": round(f1_majority, 3),
    }
