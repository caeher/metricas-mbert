"""Tests de evaluación adversarial mBERT (T-06)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.adversarial import (
    BATTERY_N,
    FAMILY_SPECS,
    adversarial_text_set,
    assert_adversarial_not_in_splits,
    assess_mitigation_gap,
    build_battery_manifest,
    export_adversarial_artifacts,
    find_train_overlaps,
    load_battery_manifest,
)
from discurso_odio.data.loaders import load_split
from discurso_odio.models.mbert_artifacts import sha256_file
from discurso_odio.paths import (
    ADVERSARIAL_144_BATTERY_MANIFEST_PATH,
    ADVERSARIAL_BATTERY_MANIFEST_PATH,
    REPORTS_METRICS,
    TEST_ADVERSARIAL_144_CSV,
    TEST_ADVERSARIAL_CSV,
)


def test_adversarial_battery_schema():
    df = load_split("adversarial")
    assert len(df) == BATTERY_N == 75
    assert set(df["grupo"].unique()) == set(FAMILY_SPECS.keys())
    for grupo, spec in FAMILY_SPECS.items():
        assert int((df["grupo"] == grupo).sum()) == spec["n"]


def test_adversarial_no_val_test_leakage():
    overlaps = assert_adversarial_not_in_splits(splits=("val", "test"))
    assert isinstance(overlaps, pd.DataFrame)


def test_battery_manifest_sha256():
    manifest = load_battery_manifest()
    assert manifest["battery_version"] == "adversarial_v1"
    assert manifest["sha256"] == sha256_file(TEST_ADVERSARIAL_CSV)
    assert manifest["n_cases"] == 75


def test_adversarial_summary_artifacts_exist():
    summary_path = REPORTS_METRICS / "mbert_adversarial_summary.json"
    if not summary_path.exists():
        pytest.skip("Ejecutar scripts/evaluate_adversarial.py primero")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    required = {
        "checkpoint",
        "battery_version",
        "battery_sha256",
        "f1_macro_diagnostic",
        "f1_amenazas",
        "threat_fn",
        "n_cases",
        "is_diagnostic_metric",
    }
    assert required.issubset(summary.keys())
    assert summary["checkpoint"] == "checkpoint-1072"
    assert summary["is_diagnostic_metric"] is True

    family_path = REPORTS_METRICS / "mbert_adversarial_family_metrics.csv"
    pred_path = REPORTS_METRICS / "mbert_adversarial_predictions.csv"
    assert family_path.exists()
    assert pred_path.exists()
    preds = pd.read_csv(pred_path)
    assert len(preds) == 75
    assert {"y_pred", "confianza", "acierto"}.issubset(preds.columns)


def test_train_overlap_excluded_for_b2():
    overlaps = find_train_overlaps()
    assert len(overlaps) <= 10
    if len(overlaps):
        assert all(overlaps["split"] == "train")


def test_mitigation_gate_logic():
    family_df = pd.DataFrame(
        [
            {"family_key": "C", "correct": 5, "n": 11},
            {"family_key": "B", "correct": 7, "n": 18},
            {"family_key": "D", "fpr": 0.5, "fpr_ci_lo": 0.3, "fpr_ci_hi": 0.7},
            {"family_key": "E", "fpr": 0.0, "fpr_ci_lo": 0.0, "fpr_ci_hi": 0.2},
        ]
    )
    result = assess_mitigation_gap(family_df, fpr_gap=0.5, ci_overlap=False)
    assert result["mitigation_required"] is True
    assert result["reasons"]


def test_export_predictions_not_in_train():
    adv = load_split("adversarial")
    y_pred = np.zeros(len(adv), dtype=int)
    y_prob = np.eye(4)[y_pred]
    pred_df = adv.copy()
    pred_df["y_true"] = adv["label"]
    pred_df["y_pred"] = y_pred
    pred_df["confianza"] = y_prob.max(axis=1)
    pred_df["acierto"] = pred_df["y_true"] == pred_df["y_pred"]
    family_df = pd.DataFrame([{"family_key": "A", "n": 20, "correct": 10}])
    summary = {"checkpoint": "test", "n_cases": 75}
    export_adversarial_artifacts(
        pred_df=pred_df,
        family_df=family_df,
        summary=summary,
        output_prefix="_pytest_adversarial_tmp",
    )
    out = REPORTS_METRICS / "_pytest_adversarial_tmp_adversarial_predictions.csv"
    assert out.exists()
    texts = set(pd.read_csv(out)["ID"].astype(str))
    train_texts = adversarial_text_set()
    assert not texts.intersection(train_texts)
    out.unlink(missing_ok=True)
    (REPORTS_METRICS / "_pytest_adversarial_tmp_adversarial_family_metrics.csv").unlink(missing_ok=True)
    (REPORTS_METRICS / "_pytest_adversarial_tmp_adversarial_summary.json").unlink(missing_ok=True)


def test_before_after_has_both_stages_if_present():
    path = REPORTS_METRICS / "mbert_adversarial_before_after.csv"
    if not path.exists():
        pytest.skip("Sin comparación B2")
    df = pd.read_csv(path)
    assert "baseline" in df["stage"].values
    assert "candidate_b2" in df["stage"].values


def test_build_battery_manifest_writes_file():
    manifest = build_battery_manifest(write=True)
    assert ADVERSARIAL_BATTERY_MANIFEST_PATH.exists()
    assert manifest["sha256"] == sha256_file(TEST_ADVERSARIAL_CSV)


def test_adversarial_144_battery_schema():
    from discurso_odio.data.adversarial_battery import BATTERY_144_N, FAMILY_SPECS_144
    from discurso_odio.evaluation.adversarial import build_battery_144_manifest, load_battery_144_manifest

    if not TEST_ADVERSARIAL_144_CSV.exists():
        pytest.skip("test_adversarial_144.csv no generado")
    df = load_split("adversarial_144")
    assert len(df) == BATTERY_144_N == 144
    for grupo, spec in FAMILY_SPECS_144.items():
        assert int((df["grupo"] == grupo).sum()) == spec["n"]
    manifest = load_battery_144_manifest()
    assert manifest["battery_version"] == "adversarial_v2_144"
    assert manifest["sha256"] == sha256_file(TEST_ADVERSARIAL_144_CSV)


def test_adversarial_144_summary_schema():
    summary_path = REPORTS_METRICS / "mbert_v1_144_adversarial_summary.json"
    if not summary_path.exists():
        pytest.skip("Ejecutar evaluate_adversarial.py --battery-144 primero")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["n_cases"] == 144
    assert summary["battery_version"] == "adversarial_v2_144"
    assert "group_g_innocuous_fp" in summary
    assert "xlmr_v1_baseline" in summary
