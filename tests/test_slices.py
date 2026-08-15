"""Tests de métricas por cortes (slices)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.indicators import normalize_indicator, resolve_slice_columns
from discurso_odio.evaluation.slices import (
    export_slice_metrics,
    slice_metrics,
    slice_metrics_to_rows,
)
from discurso_odio.paths import REPORTS_METRICS

XLMR_PREDICTIONS_PATH = REPORTS_METRICS / "xlmr_test_predictions.csv"


def _mini_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Plataforma": ["X", "X", "Facebook", "Facebook"],
            "n_jerga": [1, 0, 0, 1],
            "Presencia_Sarcasmo": ["SI", "SÍ", "NO", "si"],
            "Presencia_Ironia": ["NO", "Sí", "NO", "YES"],
            "Contexto_Politico": ["NO", "SI", "tal vez", "NO"],
        }
    )


def test_resolve_slice_columns_on_test_split():
    test = load_split("test")
    resolved = resolve_slice_columns(test)
    assert resolved["Presencia_Sarcasmo"] == "Presencia_Sarcasmo"
    assert resolved["Presencia_Ironia"] == "Presencia_Ironia"
    assert resolved["Contexto_Politico"] == "Contexto_Politico"


def test_resolve_slice_columns_alias():
    df = pd.DataFrame({"sarcasmo": ["SI"], "ironia": ["NO"], "contexto_politico": ["SI"]})
    resolved = resolve_slice_columns(df)
    assert resolved["Presencia_Sarcasmo"] == "sarcasmo"
    assert resolved["Presencia_Ironia"] == "ironia"
    assert resolved["Contexto_Politico"] == "contexto_politico"


def test_normalize_indicator_accent_variants():
    series = pd.Series(["SI", "SÍ", "sí", "Sí", "NO", "no", "", "NA", "2", "tal vez"])
    normalized = normalize_indicator(series)
    assert normalized.tolist()[:4] == [True, True, True, True]
    assert normalized.tolist()[4:6] == [False, False]
    assert normalized.tolist()[6:].count(pd.NA) == 4


def test_unexpected_values_excluded_from_both_slices():
    df = _mini_df()
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    results = slice_metrics(df, y_true, y_pred, include_empty_slices=True)

    assert results["contexto_politico_si"]["n"] == 1
    assert results["contexto_politico_no"]["n"] == 2
    # "tal vez" no entra en ningún corte


def test_si_and_no_slices_have_observations():
    df = _mini_df()
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    results = slice_metrics(df, y_true, y_pred)

    assert results["sarcasmo_si"]["n"] == 3
    assert results["sarcasmo_no"]["n"] == 1
    assert results["ironia_si"]["n"] == 2
    assert results["ironia_no"]["n"] == 2


def test_empty_slice_has_nan_metrics():
    df = pd.DataFrame(
        {
            "Plataforma": ["X"],
            "Presencia_Sarcasmo": ["NO"],
            "Presencia_Ironia": ["NO"],
            "Contexto_Politico": ["NO"],
        }
    )
    y_true = np.array([0])
    y_pred = np.array([0])
    results = slice_metrics(df, y_true, y_pred, include_empty_slices=True)

    assert results["sarcasmo_si"]["n"] == 0
    assert pd.isna(results["sarcasmo_si"]["f1_macro"])
    assert pd.isna(results["sarcasmo_si"]["accuracy"])


def test_bootstrap_ci_only_when_n_large_enough():
    df = pd.DataFrame({"Plataforma": ["X"] * 35, "Presencia_Sarcasmo": ["NO"] * 35})
    y_true = np.tile([0, 1, 2, 3], 9)[:35]
    y_pred = y_true.copy()
    results = slice_metrics(df, y_true, y_pred)
    entry = results["plataforma_X"]
    assert entry["n"] == 35
    assert pd.notna(entry["f1_ci_lo"])
    assert pd.notna(entry["f1_ci_hi"])


def test_export_slice_metrics_writes_csv_and_json(tmp_path):
    df = _mini_df()
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2])
    slices = slice_metrics(df, y_true, y_pred)
    rows = slice_metrics_to_rows(
        slices,
        model="mbert",
        checkpoint="models/mbert-sv",
        split="test.csv",
        n_test=len(df),
    )
    slice_df = pd.DataFrame(rows)
    csv_path, json_path = export_slice_metrics(
        slice_df,
        tmp_path,
        extra_metadata={"xlmr_status": "predictions_not_found"},
    )

    assert csv_path.exists()
    assert json_path.exists()
    loaded = pd.read_csv(csv_path)
    assert set(loaded.columns) >= {"slice", "n", "f1_macro", "accuracy", "model", "checkpoint"}
    assert (loaded["n"] > 0).all()

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["xlmr_status"] == "predictions_not_found"
    assert "created_at" in meta
    assert len(meta["rows"]) == len(loaded)


def test_slice_metrics_on_real_test_has_indicator_slices():
    test = load_split("test")
    y_true = test["label"].values
    y_pred = y_true.copy()
    results = slice_metrics(test, y_true, y_pred)

    assert results["sarcasmo_si"]["n"] > 0
    assert results["ironia_si"]["n"] > 0
    assert results["contexto_politico_si"]["n"] > 0


@pytest.mark.skipif(
    not XLMR_PREDICTIONS_PATH.exists(),
    reason="Predicciones XLM-R auditadas no disponibles",
)
def test_xlmr_slices_recalc_not_hardcoded():
    """Documenta que las cifras de auditoría XLM-R se recalculan, no se transcriben."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compute_slice_metrics import compute_all

    slice_df = compute_all(include_xlmr=True)
    xlmr = slice_df[slice_df["model"] == "xlm-roberta"]
    assert not xlmr.empty
    sarcasmo_si = xlmr[xlmr["slice"] == "sarcasmo_si"].iloc[0]
    sarcasmo_no = xlmr[xlmr["slice"] == "sarcasmo_no"].iloc[0]
    assert sarcasmo_si["n"] > 0
    assert sarcasmo_no["n"] > 0
    assert pd.notna(sarcasmo_si["f1_macro"])
    assert pd.notna(sarcasmo_no["f1_macro"])
