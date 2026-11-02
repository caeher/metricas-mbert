"""Tests production_slo_v1 scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from discurso_odio.evaluation.production_slo import (
    _fpr_dialectal_gap_from_probe,
    evaluate_production_slo,
    load_production_slo,
)
from discurso_odio.paths import REPORTS_METRICS


@pytest.fixture
def slo_contract(tmp_path: Path) -> Path:
    path = tmp_path / "production_slo_v1.json"
    path.write_text(
        json.dumps(
            {
                "slo_id": "test_slo",
                "target_operational_score": 0.9,
                "platform_traffic_weights": {
                    "plataforma_Facebook": 0.5,
                    "plataforma_X": 0.5,
                },
                "min_slice_f1_macro": {
                    "plataforma_Facebook": 0.75,
                    "plataforma_X": 0.85,
                    "con_jerga": 0.77,
                },
                "min_class_recall": {"Amenazas/Violencia": 0.8},
                "constraints": {
                    "adv75_min_correct": 45,
                    "adv75_n": 75,
                    "fpr_dialectal_gap_max": 0.17,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_production_slo():
    slo = load_production_slo(REPORTS_METRICS / "production_slo_v1.json")
    assert slo["slo_id"] == "production_slo_v1"
    assert slo["target_operational_score"] == 0.9


@pytest.mark.skipif(
    not (REPORTS_METRICS / "mbert_test_metrics.csv").exists(),
    reason="Faltan métricas oficiales exportadas",
)
def test_evaluate_production_slo_on_existing_metrics(slo_contract: Path, tmp_path: Path, monkeypatch):
    metrics = pd.read_csv(REPORTS_METRICS / "mbert_test_metrics.csv")
    slices = pd.read_csv(REPORTS_METRICS / "slice_metrics.csv")
    monkeypatch.setattr(
        "discurso_odio.evaluation.production_slo.REPORTS_METRICS",
        tmp_path,
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tmp_path / "mbert_test_metrics.csv", index=False)
    slices.to_csv(tmp_path / "slice_metrics.csv", index=False)
    (tmp_path / "dialect_bias_probe.csv").write_text("fpr_gap\n0.10\n", encoding="utf-8")

    result = evaluate_production_slo(
        model_dir=None,
        slo_path=slo_contract,
        run_adversarial=False,
        use_existing_metrics=False,
    )
    assert result["operational_score"] is not None
    assert result["operational_score"] < 0.9
    assert result["slo_pass"] is False
    assert "weighted_platform_f1_macro" in result["components"]


def test_fpr_dialectal_gap_from_probe_rows():
    probe = pd.DataFrame(
        [
            {
                "model": "mbert",
                "checkpoint": "checkpoint-1",
                "group": "con_jerga",
                "fpr": 0.34,
            },
            {
                "model": "mbert",
                "checkpoint": "checkpoint-1",
                "group": "sin_jerga",
                "fpr": 0.12,
            },
        ]
    )
    gap = _fpr_dialectal_gap_from_probe(probe, checkpoint="checkpoint-1")
    assert gap is not None
    assert abs(gap - 0.22) < 1e-9
