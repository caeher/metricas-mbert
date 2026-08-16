"""Tests de McNemar pareado y validación de vectores de predicciones."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.mcnemar import mcnemar_conclusion, run_mcnemar_tests
from discurso_odio.evaluation.metrics import mcnemar_test
from discurso_odio.evaluation.predictions import validate_prediction_vector
from discurso_odio.evaluation.report_consistency import check_mcnemar_metrics
from discurso_odio.paths import REPORTS_METRICS

XLMR_PREDICTIONS_PATH = REPORTS_METRICS / "xlmr_test_predictions.csv"
MCNEMAR_PATH = REPORTS_METRICS / "mcnemar_tests.csv"


def _mini_test(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"ID": list(range(100, 100 + n)), "label": [0, 1, 2, 3][:n]})


def _preds_from_ids(ids: list[int], labels: list[int], col: str = "y_pred_xlmr") -> pd.DataFrame:
    return pd.DataFrame({"id": ids, col: labels})


def test_validate_rejects_wrong_length():
    test = _mini_test()
    preds = _preds_from_ids([100, 101], [0, 1])
    with pytest.raises(ValueError, match="se esperaban 4"):
        validate_prediction_vector(test, preds, "y_pred_xlmr", "XLM-R")


def test_validate_rejects_missing_ids():
    test = _mini_test()
    preds = _preds_from_ids([100, 101, 102, 999], [0, 1, 2, 3])
    with pytest.raises(ValueError, match="sin ID correspondiente"):
        validate_prediction_vector(test, preds, "y_pred_xlmr", "XLM-R")


def test_validate_rejects_out_of_range_labels():
    test = _mini_test()
    preds = _preds_from_ids([100, 101, 102, 103], [0, 1, 4, 3])
    with pytest.raises(ValueError, match="fuera del rango"):
        validate_prediction_vector(test, preds, "y_pred_xlmr", "XLM-R")


def test_validate_rejects_non_integer_labels():
    test = _mini_test()
    preds = pd.DataFrame({"id": [100, 101, 102, 103], "y_pred_xlmr": [0, 1.5, 2, 3]})
    with pytest.raises(ValueError, match="enteras"):
        validate_prediction_vector(test, preds, "y_pred_xlmr", "XLM-R")


def test_validate_accepts_id_aligned_vector():
    test = _mini_test()
    preds = _preds_from_ids([103, 100, 102, 101], [3, 0, 2, 1])
    y_pred = validate_prediction_vector(test, preds, "y_pred_xlmr", "XLM-R")
    np.testing.assert_array_equal(y_pred, [0, 1, 2, 3])


def test_mcnemar_rejects_length_mismatch():
    y_true = np.array([0, 1, 2])
    pred_a = np.array([0, 1, 2])
    pred_b = np.array([0, 1])
    with pytest.raises(ValueError, match="Longitudes incompatibles"):
        mcnemar_test(y_true, pred_a, pred_b)


def test_mcnemar_known_counts():
    y_true = np.array([0, 1, 2, 3, 0, 1])
    pred_a = np.array([0, 1, 2, 2, 1, 1])  # correct on 0,1,2
    pred_b = np.array([0, 0, 2, 3, 0, 1])  # correct on 0,2,3,0,1
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["b"] == 1  # a correct, b wrong at index 1
    assert result["c"] == 2  # a wrong, b correct at indices 3 and 4


def test_mcnemar_no_discordant_pairs():
    y_true = np.array([0, 1, 2, 3])
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_conclusion_text():
    assert "no se rechaza" in mcnemar_conclusion(0.20, 0.05)
    assert "significativa" in mcnemar_conclusion(0.01, 0.05)


def test_run_mcnemar_export_schema(tmp_path, monkeypatch):
    from discurso_odio.data import loaders

    test = loaders.load_split("test")
    n = len(test)
    y_true = test["label"].values

    xlmr_csv = tmp_path / "xlmr_test_predictions.csv"
    xlmr_df = pd.DataFrame({"id": test["ID"].values, "y_pred_xlmr": y_true})
    xlmr_df.to_csv(xlmr_csv, index=False)

    mbert_csv = tmp_path / "mbert_test_predictions.csv"
    mbert_df = pd.DataFrame({"id": test["ID"].values, "y_pred_mbert": y_true})
    mbert_df.to_csv(mbert_csv, index=False)

    out_csv = tmp_path / "mcnemar_tests.csv"

    import discurso_odio.evaluation.mcnemar as mcnemar_mod
    import discurso_odio.evaluation.predictions as pred_mod

    monkeypatch.setattr(pred_mod, "XLMR_PREDICTIONS_PATH", xlmr_csv)
    monkeypatch.setattr(pred_mod, "MBERT_PREDICTIONS_PATH", mbert_csv)
    monkeypatch.setattr(mcnemar_mod, "XLMR_PREDICTIONS_PATH", xlmr_csv)
    monkeypatch.setattr(mcnemar_mod, "MBERT_PREDICTIONS_PATH", mbert_csv)
    monkeypatch.setattr(mcnemar_mod, "MCNEMAR_PATH", out_csv)

    df = run_mcnemar_tests(require_xlmr=True, output_path=out_csv)
    xlmr_row = df[df["model_b"] == "xlm-roberta"].iloc[0]
    assert xlmr_row["model_a"] == "mbert"
    assert int(xlmr_row["n"]) == n
    assert float(xlmr_row["alpha"]) == 0.05
    assert "conclusion" in xlmr_row
    assert "checkpoint_a" in xlmr_row
    assert "checkpoint_b" in xlmr_row
    assert xlmr_row["p_value"] == 1.0


def test_mcnemar_latex_consistency(tmp_path):
    mcnemar_csv = tmp_path / "mcnemar_tests.csv"
    pd.DataFrame(
        [
            {
                "model_a": "mbert",
                "model_b": "xlm-roberta",
                "checkpoint_a": "models/mbert-sv",
                "checkpoint_b": "XLMR/02_modelo_prototipo/modelo_entrenado",
                "statistic": 0.4098,
                "p_value": 0.200031,
                "b": 25,
                "c": 36,
                "n": 460,
                "alpha": 0.05,
                "significant": False,
                "conclusion": "no se rechaza H0 (alpha=0.05)",
            }
        ]
    ).to_csv(mcnemar_csv, index=False)

    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "tab_mcnemar.tex").write_text("0.200031\n", encoding="utf-8")
    (generated / "mcnemar_prose.tex").write_text("b=25, c=36, p=0.200031\n", encoding="utf-8")

    errors = check_mcnemar_metrics(mcnemar_csv, generated)
    assert errors == []


@pytest.mark.skipif(
    not MCNEMAR_PATH.exists() or not XLMR_PREDICTIONS_PATH.exists(),
    reason="Artefactos McNemar/XLM-R no generados",
)
def test_real_mcnemar_has_xlmr_row():
    df = pd.read_csv(MCNEMAR_PATH)
    xlmr = df[df["model_b"] == "xlm-roberta"]
    assert not xlmr.empty
    row = xlmr.iloc[0]
    assert int(row["n"]) == 460
    assert float(row["alpha"]) == 0.05
    assert {"statistic", "p_value", "b", "c", "conclusion", "checkpoint_a", "checkpoint_b"} <= set(
        row.index
    )
