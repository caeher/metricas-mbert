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


def test_trivial_baselines():
    counts = {0: 100, 1: 200, 2: 150, 3: 50}
    baselines = trivial_baselines(counts)
    assert baselines["azar_estratificado"] == 0.25
