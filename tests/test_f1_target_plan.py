"""Tests plan F1 0.80 — loaders, pérdidas, umbrales, error analysis."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import (
    compute_sample_loss_weights,
    default_facebook_boost_weights,
    filter_train_by_synthetic_policy,
    load_split,
    load_train_excluding_adversarial_overlap,
    summarize_train_platform_class_weights,
)
from discurso_odio.evaluation.error_analysis import analyze_errors, merge_test_predictions
from discurso_odio.evaluation.thresholds import apply_class_thresholds, grid_search_thresholds
from discurso_odio.models.losses import combined_ordinal_loss, focal_loss
from discurso_odio.paths import REPORTS_METRICS
import torch


def test_filter_synthetic_policies():
    train = load_split("train")
    real_only = filter_train_by_synthetic_policy(train, "real_only")
    assert len(real_only) == 2144
    no_heavy = filter_train_by_synthetic_policy(train, "b2_no_template_heavy")
    assert len(no_heavy) < len(train)
    assert len(no_heavy) > len(real_only)


def test_synthetic_downweight_fraction():
    train = load_train_excluding_adversarial_overlap()
    weights = compute_sample_loss_weights(train, synthetic_effective_fraction=0.35)
    synth_mask = train["origen"] != "real"
    non_threat = ~train["origen"].isin({"sintetico_amenaza_ofuscada", "sintetico_emoji_amenaza"})
    assert abs(float(weights[synth_mask & non_threat].mean()) - 0.35) < 1e-6


def test_platform_class_boost_increases_facebook_odio():
    train = load_train_excluding_adversarial_overlap()
    boost = default_facebook_boost_weights()
    weights = compute_sample_loss_weights(train, platform_class_boost=boost)
    fb_odio = (train["Plataforma"] == "Facebook") & (train["label"] == 2) & (train["origen"] == "real")
    assert float(weights[fb_odio].iloc[0]) == pytest.approx(2.5)


def test_summarize_train_platform_class_weights():
    summary = summarize_train_platform_class_weights()
    assert summary["n_train"] == 3293
    fb_odio = [r for r in summary["rows"] if r["platform"] == "Facebook" and r["label"] == 2][0]
    assert fb_odio["n_real"] == 141
    assert fb_odio["n_synthetic"] > 0


def test_merge_test_predictions():
    merged = merge_test_predictions()
    assert len(merged) == 460
    assert "y_pred" in merged.columns


def test_analyze_errors_structure():
    merged = merge_test_predictions()
    report = analyze_errors(merged)
    assert report["n_test"] == 460
    assert report["n_errors"] == int((merged["label"] != merged["y_pred"]).sum())
    assert "by_platform" in report
    assert "Facebook" in report["by_platform"]
    assert len(report["recommendations"]) >= 1


def test_error_analysis_export_if_predictions_exist():
    if not (REPORTS_METRICS / "mbert_test_predictions.csv").exists():
        pytest.skip("Sin predicciones oficiales")
    from discurso_odio.evaluation.error_analysis import export_error_analysis

    report = export_error_analysis()
    assert (REPORTS_METRICS / "error_analysis_b2.json").exists()
    payload = json.loads((REPORTS_METRICS / "error_analysis_b2.json").read_text(encoding="utf-8"))
    assert payload["f1_macro"] == pytest.approx(report["f1_macro"], rel=1e-4)


def test_focal_loss_shape():
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    loss = focal_loss(logits, labels, gamma=2.0)
    assert loss.shape == (8,)


def test_ordinal_combined_loss():
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    loss = combined_ordinal_loss(logits, labels, ordinal_weight=0.3)
    assert loss.shape == (8,)


def test_threshold_grid_improves_or_matches_argmax():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 4, size=200)
    y_prob = np.eye(4)[y_true] * 0.6 + rng.random((200, 4)) * 0.1
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    _, best_f1 = grid_search_thresholds(y_true, y_prob, grid=[0.2, 0.25, 0.3])
    baseline = float(
        __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
            y_true, np.argmax(y_prob, axis=1), average="macro", zero_division=0
        )
    )
    assert best_f1 >= baseline - 1e-6


def test_evaluate_gates_existing_metrics():
    from discurso_odio.evaluation.f1_gates import evaluate_gates

    if not (REPORTS_METRICS / "mbert_test_metrics.csv").exists():
        pytest.skip("Sin métricas oficiales")
    gates = evaluate_gates(None, use_existing_metrics=True, run_official=False)
    assert "test_f1_macro" in gates
    assert gates["target_f1_macro"] == 0.80
    assert gates["primary_pass"] is (gates["test_f1_macro"] >= 0.80)
