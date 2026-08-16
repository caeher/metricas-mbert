"""Pruebas de métrica oficial mBERT y consistencia del informe (T-04)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.official_mbert import (
    MBERT_METRICS_PATH,
    MBERT_OFFICIAL_RUN_PATH,
    MBERT_PREDICTIONS_PATH,
    MBERT_TRACEABILITY_CSV,
    build_traceability_rows,
)
from discurso_odio.evaluation.report_consistency import (
    assert_mbert_official_consistency,
    check_mbert_official_metrics,
)
from discurso_odio.models.mbert_artifacts import OFFICIAL_F1_MACRO
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS

GENERATED_DIR = PROJECT_ROOT / "reports" / "latex" / "generated"
ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = ROOT / ".gitignore"


def test_gitignore_tracks_official_metric_artifacts():
    text = GITIGNORE.read_text(encoding="utf-8")
    for pattern in (
        "!reports/metrics/mbert_test_metrics.csv",
        "!reports/metrics/mbert_test_probabilities.csv",
        "!reports/metrics/mbert_confusion_matrix.csv",
        "!reports/metrics/mbert_calibration.csv",
        "!reports/metrics/mbert_official_run.json",
        "!reports/metrics/mbert_f1_traceability.csv",
    ):
        assert pattern in text


@pytest.mark.skipif(not MBERT_METRICS_PATH.exists(), reason="Falta evaluación oficial")
def test_official_metrics_round_to_07777():
    row = pd.read_csv(MBERT_METRICS_PATH).iloc[0]
    assert round(float(row["f1_macro"]), 4) == OFFICIAL_F1_MACRO
    assert len(pd.read_csv(MBERT_PREDICTIONS_PATH)) == 460


@pytest.mark.skipif(not MBERT_OFFICIAL_RUN_PATH.exists(), reason="Falta official run")
def test_official_run_aligns_with_metrics():
    with open(MBERT_OFFICIAL_RUN_PATH, encoding="utf-8") as handle:
        run = json.load(handle)
    row = pd.read_csv(MBERT_METRICS_PATH).iloc[0]
    assert run["official_f1_macro"] == OFFICIAL_F1_MACRO
    assert run["test_n"] == 460
    assert abs(float(run["f1_macro_raw"]) - float(row["f1_macro"])) < 1e-9
    assert run["artifacts"]["metrics_csv"] == "reports/metrics/mbert_test_metrics.csv"


def test_traceability_marks_audit_figure_not_reproducible():
    rows = build_traceability_rows(official_f1_raw=OFFICIAL_F1_MACRO)
    audit = [r for r in rows if r["f1_macro_published"] == 0.7862]
    assert len(audit) == 1
    assert audit[0]["status"] == "not_reproducible"
    official = [r for r in rows if r["status"] == "official"]
    assert len(official) == 1
    assert official[0]["f1_macro_published"] == OFFICIAL_F1_MACRO


@pytest.mark.skipif(not MBERT_TRACEABILITY_CSV.exists(), reason="Falta trazabilidad")
def test_traceability_csv_has_official_row():
    df = pd.read_csv(MBERT_TRACEABILITY_CSV)
    official = df[df["status"] == "official"]
    assert len(official) == 1
    assert float(official.iloc[0]["f1_macro_published"]) == OFFICIAL_F1_MACRO


@pytest.mark.skipif(
    not (MBERT_METRICS_PATH.exists() and (GENERATED_DIR / "tab_mbert_global.tex").exists()),
    reason="Falta evaluación o LaTeX generado",
)
def test_latex_global_table_matches_csv():
    row = pd.read_csv(MBERT_METRICS_PATH).iloc[0]
    tex = (GENERATED_DIR / "tab_mbert_global.tex").read_text(encoding="utf-8")
    assert f"{float(row['f1_macro']):.4f}" in tex
    assert "0.7862" not in tex
    assert "0.778" not in tex


@pytest.mark.skipif(
    not MBERT_METRICS_PATH.exists(),
    reason="Falta evaluación oficial",
)
def test_report_consistency_checker_passes():
    required = [
        "tab_mbert_global.tex",
        "mbert_official_prose.tex",
        "mbert_intro_prose.tex",
        "mbert_conclusions_prose.tex",
        "mbert_benchmark_prose.tex",
        "tab_mbert_traceability.tex",
    ]
    for name in required:
        if not (GENERATED_DIR / name).exists():
            pytest.skip(f"Falta {name}; ejecutar generate_latex_report.py")
    errors = check_mbert_official_metrics(MBERT_METRICS_PATH, GENERATED_DIR)
    assert not errors, "\n".join(e.message for e in errors)
    assert_mbert_official_consistency(MBERT_METRICS_PATH, GENERATED_DIR)
