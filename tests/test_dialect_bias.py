"""Tests de la sonda de sesgo dialectal."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.slices import (
    JERGA_DEFINITION,
    dialect_bias_probe,
    export_dialect_bias,
)
from discurso_odio.paths import REPORTS_METRICS

XLMR_PREDICTIONS_PATH = REPORTS_METRICS / "xlmr_test_predictions.csv"


def _make_synthetic_df(
    n_con: int,
    fp_con: int,
    n_sin: int,
    fp_sin: int,
    total_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Construye un mini test con control exacto de FP por grupo de jerga."""
    total_rows = total_rows or (n_con + n_sin + 10)
    rows = []
    y_true: list[int] = []
    y_pred: list[int] = []

    # No tóxico con jerga
    for i in range(n_con):
        rows.append({"n_jerga": 1})
        y_true.append(0)
        y_pred.append(1 if i < fp_con else 0)

    # No tóxico sin jerga
    for i in range(n_sin):
        rows.append({"n_jerga": 0})
        y_true.append(0)
        y_pred.append(1 if i < fp_sin else 0)

    # Relleno: tóxicos y otros para llegar a total_rows
    while len(rows) < total_rows:
        rows.append({"n_jerga": 0})
        y_true.append(1)
        y_pred.append(1)

    return pd.DataFrame(rows), np.array(y_true), np.array(y_pred)


def test_dialect_bias_audit_values():
    """Recálculo de auditoría: 13/29 con jerga y 15/94 sin jerga."""
    df, y_true, y_pred = _make_synthetic_df(n_con=29, fp_con=13, n_sin=94, fp_sin=15)
    result = dialect_bias_probe(df, y_true, y_pred, model="xlm-roberta")

    con = result[result["group"] == "con_jerga"].iloc[0]
    sin = result[result["group"] == "sin_jerga"].iloc[0]

    assert con["n"] == 29
    assert con["fp"] == 13
    assert sin["n"] == 94
    assert sin["fp"] == 15
    assert con["fpr"] == pytest.approx(13 / 29, rel=1e-6)
    assert sin["fpr"] == pytest.approx(15 / 94, rel=1e-6)
    assert con["fpr"] == pytest.approx(0.4483, rel=1e-3)
    assert sin["fpr"] == pytest.approx(0.1596, rel=1e-3)


def test_dialect_bias_empty_group_nan_fpr():
    df, y_true, y_pred = _make_synthetic_df(n_con=0, fp_con=0, n_sin=5, fp_sin=2)
    result = dialect_bias_probe(df, y_true, y_pred)

    con = result[result["group"] == "con_jerga"].iloc[0]
    sin = result[result["group"] == "sin_jerga"].iloc[0]

    assert con["n"] == 0
    assert con["fp"] == 0
    assert pd.isna(con["fpr"])
    assert pd.isna(con["fpr_ci_lo"])
    assert pd.isna(con["fpr_ci_hi"])
    assert sin["fpr"] == pytest.approx(2 / 5)


def test_dialect_bias_wilson_ci_only_when_n_positive():
    df, y_true, y_pred = _make_synthetic_df(n_con=10, fp_con=3, n_sin=0, fp_sin=0)
    result = dialect_bias_probe(df, y_true, y_pred)

    con = result[result["group"] == "con_jerga"].iloc[0]
    sin = result[result["group"] == "sin_jerga"].iloc[0]

    assert con["fpr_ci_lo"] < con["fpr"] < con["fpr_ci_hi"]
    assert pd.isna(sin["fpr_ci_lo"])
    assert pd.isna(sin["fpr_ci_hi"])


def test_dialect_bias_missing_n_jerga_returns_empty():
    df = pd.DataFrame({"texto_modelo": ["hola", "adios"]})
    y_true = np.array([0, 1])
    y_pred = np.array([1, 1])
    result = dialect_bias_probe(df, y_true, y_pred)
    assert result.empty


def test_export_dialect_bias_writes_csv_and_json(tmp_path):
    df, y_true, y_pred = _make_synthetic_df(n_con=3, fp_con=1, n_sin=4, fp_sin=2)
    probe = dialect_bias_probe(df, y_true, y_pred, model="mbert", checkpoint="models/mbert-sv")

    csv_path, json_path = export_dialect_bias(
        probe,
        tmp_path,
        extra_metadata={"xlmr_status": "predictions_not_found"},
    )

    assert csv_path.exists()
    assert json_path.exists()

    loaded = pd.read_csv(csv_path)
    assert set(loaded.columns) >= {"model", "group", "n", "fp", "fpr", "run_id", "checkpoint"}
    assert loaded["jerga_definition"].iloc[0] == JERGA_DEFINITION

    import json

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["xlmr_status"] == "predictions_not_found"
    assert len(meta["rows"]) == 2
    assert "created_at" in meta


@pytest.mark.skipif(
    not XLMR_PREDICTIONS_PATH.exists(),
    reason="Predicciones XLM-R auditadas no disponibles",
)
def test_xlmr_integration_audit_values():
    """Valida 15/94 y 13/29 si existe el CSV de predicciones XLM-R."""
    from discurso_odio.data.loaders import load_split

    test = load_split("test")
    preds = pd.read_csv(XLMR_PREDICTIONS_PATH)
    if "id" in preds.columns and "ID" in test.columns:
        merged = test[["ID", "label", "n_jerga"]].merge(
            preds, left_on="ID", right_on="id", how="left"
        )
        y_pred = merged["y_pred_xlmr"].to_numpy(dtype=int)
        y_true = merged["label"].to_numpy()
        df = merged
    else:
        y_pred = preds["y_pred_xlmr"].to_numpy(dtype=int)
        y_true = test["label"].values
        df = test

    result = dialect_bias_probe(df, y_true, y_pred, model="xlm-roberta")
    con = result[result["group"] == "con_jerga"].iloc[0]
    sin = result[result["group"] == "sin_jerga"].iloc[0]

    assert con["fp"] == 13 and con["n"] == 29
    assert sin["fp"] == 15 and sin["n"] == 94
