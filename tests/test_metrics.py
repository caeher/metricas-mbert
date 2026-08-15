"""Tests de métricas."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.metrics import (
    compute_all_metrics,
    ordinal_mae,
    trivial_baselines,
)


def test_ordinal_mae_perfect():
    y = np.array([0, 1, 2, 3])
    assert ordinal_mae(y, y) == 0.0


def test_ordinal_mae_one_off():
    y_true = np.array([0, 3])
    y_pred = np.array([1, 2])
    assert ordinal_mae(y_true, y_pred) == 1.0


def test_compute_all_metrics_keys():
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 2, 1, 1])
    metrics = compute_all_metrics(y_true, y_pred)
    assert "f1_macro" in metrics
    assert "ordinal_mae" in metrics
    assert "mcc" in metrics


def test_trivial_baselines_audit_test_distribution():
    """Distribución real de test: 159/460 en clase mayoritaria (label 1)."""
    counts = {0: 123, 1: 159, 2: 95, 3: 83}
    baselines = trivial_baselines(counts)
    assert baselines["clase_mayoritaria"] == 0.128
    assert baselines["azar_estratificado"] == 0.25
    assert baselines["azar_uniforme"] == 0.25


def test_trivial_baselines_second_distribution():
    counts = {0: 100, 1: 200, 2: 150, 3: 50}
    baselines = trivial_baselines(counts)
    assert baselines["azar_estratificado"] == 0.25
    assert baselines["clase_mayoritaria"] == 0.143


def test_trivial_baselines_divides_by_num_classes_not_present_labels():
    """F1-macro usa k=4 aunque falten etiquetas en el dict de conteos."""
    counts = {0: 100, 1: 200}
    baselines = trivial_baselines(counts)
    assert baselines["clase_mayoritaria"] == round((2 * (2 / 3) / (2 / 3 + 1)) / 4, 3)
