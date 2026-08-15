"""Tests de métricas por plataforma y consistencia del informe (T-03)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from discurso_odio.evaluation.report_consistency import (
    assert_platform_consistency,
    check_platform_metrics,
)
from discurso_odio.evaluation.slices import (
    export_slice_metrics,
    platform_rows,
    slice_metrics,
    slice_metrics_to_rows,
)
from discurso_odio.paths import REPORTS_METRICS

XLMR_PREDICTIONS_PATH = REPORTS_METRICS / "xlmr_test_predictions.csv"
GENERATED_DIR = Path(__file__).resolve().parents[1] / "reports" / "latex" / "generated"


def _synthetic_platform_predictions(
    f1_x: float,
    f1_fb: float,
    n_x: int = 228,
    n_fb: int = 232,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Construye predicciones que producen F1-macro aproximado por plataforma."""
    rows = []
    y_true: list[int] = []
    y_pred: list[int] = []

    def _add_block(platform: str, n: int, target_f1: float):
        correct = int(round(target_f1 * n))
        correct = min(max(correct, 0), n)
        for i in range(n):
            rows.append({"Plataforma": platform})
            label = i % 4
            y_true.append(label)
            if i < correct:
                y_pred.append(label)
            else:
                y_pred.append((label + 1) % 4)

    _add_block("X", n_x, f1_x)
    _add_block("Facebook", n_fb, f1_fb)
    return pd.DataFrame(rows), np.array(y_true), np.array(y_pred)


def test_export_slice_metrics_merge_preserves_other_models(tmp_path):
    xlmr_row = pd.DataFrame(
        [
            {
                "run_id": "x1",
                "model": "xlm-roberta",
                "checkpoint": "preds.csv",
                "split": "test.csv",
                "n_test": 460,
                "slice": "plataforma_X",
                "n": 228,
                "f1_macro": 0.867,
                "accuracy": 0.87,
                "f1_ci_lo": float("nan"),
                "f1_ci_hi": float("nan"),
            }
        ]
    )
    export_slice_metrics(xlmr_row, tmp_path, merge=False)

    mbert_slices = {
        "plataforma_X": {"n": 228, "f1_macro": 0.8386, "accuracy": 0.8421, "f1_ci_lo": float("nan"), "f1_ci_hi": float("nan")},
        "plataforma_Facebook": {"n": 232, "f1_macro": 0.6727, "accuracy": 0.7241, "f1_ci_lo": float("nan"), "f1_ci_hi": float("nan")},
    }
    mbert_rows = slice_metrics_to_rows(
        mbert_slices,
        model="mbert",
        checkpoint="models/mbert-sv",
        split="test.csv",
        n_test=460,
    )
    export_slice_metrics(pd.DataFrame(mbert_rows), tmp_path, merge=True)

    loaded = pd.read_csv(tmp_path / "slice_metrics.csv")
    assert "xlm-roberta" in loaded["model"].values
    assert "mbert" in loaded["model"].values
    xlmr_x = loaded[(loaded["model"] == "xlm-roberta") & (loaded["slice"] == "plataforma_X")].iloc[0]
    assert xlmr_x["f1_macro"] == pytest.approx(0.867, rel=1e-3)


def test_platform_rows_filters_only_platform_slices():
    df = pd.DataFrame(
        {
            "model": ["mbert", "mbert", "mbert"],
            "slice": ["plataforma_X", "plataforma_Facebook", "ironia_si"],
            "n": [228, 232, 59],
            "f1_macro": [0.84, 0.67, 0.52],
        }
    )
    plat = platform_rows(df)
    assert len(plat) == 2
    assert set(plat["slice"]) == {"plataforma_X", "plataforma_Facebook"}


def test_xlmr_platform_metrics_not_mbert_values():
    df, y_true, y_pred = _synthetic_platform_predictions(0.867, 0.715)
    results = slice_metrics(df, y_true, y_pred)
    assert results["plataforma_X"]["f1_macro"] == pytest.approx(0.867, abs=0.02)
    assert results["plataforma_Facebook"]["f1_macro"] == pytest.approx(0.715, abs=0.02)
    assert results["plataforma_X"]["f1_macro"] != pytest.approx(0.8386, abs=0.01)
    assert results["plataforma_Facebook"]["f1_macro"] != pytest.approx(0.6727, abs=0.01)


def test_consistency_checker_fails_on_mbert_values_in_xlmr_prose(tmp_path):
    slice_df = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "model": "mbert",
                "checkpoint": "models/mbert-sv",
                "split": "test.csv",
                "n_test": 460,
                "slice": "plataforma_X",
                "n": 228,
                "f1_macro": 0.838632,
                "accuracy": 0.842105,
                "f1_ci_lo": float("nan"),
                "f1_ci_hi": float("nan"),
            },
            {
                "run_id": "r1",
                "model": "mbert",
                "checkpoint": "models/mbert-sv",
                "split": "test.csv",
                "n_test": 460,
                "slice": "plataforma_Facebook",
                "n": 232,
                "f1_macro": 0.672663,
                "accuracy": 0.724138,
                "f1_ci_lo": float("nan"),
                "f1_ci_hi": float("nan"),
            },
            {
                "run_id": "r2",
                "model": "xlm-roberta",
                "checkpoint": "preds.csv",
                "split": "test.csv",
                "n_test": 460,
                "slice": "plataforma_X",
                "n": 228,
                "f1_macro": 0.867,
                "accuracy": 0.87,
                "f1_ci_lo": float("nan"),
                "f1_ci_hi": float("nan"),
            },
            {
                "run_id": "r2",
                "model": "xlm-roberta",
                "checkpoint": "preds.csv",
                "split": "test.csv",
                "n_test": 460,
                "slice": "plataforma_Facebook",
                "n": 232,
                "f1_macro": 0.715,
                "accuracy": 0.72,
                "f1_ci_lo": float("nan"),
                "f1_ci_hi": float("nan"),
            },
        ]
    )
    slice_csv = tmp_path / "slice_metrics.csv"
    slice_df.to_csv(slice_csv, index=False)

    gen = tmp_path / "generated"
    gen.mkdir()
    (gen / "tab_platform.tex").write_text(
        "\\begin{tabular}{lllrrr}\n"
        "mBERT & X & ckpt & 228 & 0.8386 & 0.8421 \\\\\n"
        "mBERT & Facebook & ckpt & 232 & 0.6727 & 0.7241 \\\\\n"
        "XLM-RoBERTa & X & ckpt & 228 & 0.8670 & 0.8700 \\\\\n"
        "XLM-RoBERTa & Facebook & ckpt & 232 & 0.7150 & 0.7200 \\\\\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )
    (gen / "platform_prose_mbert.tex").write_text(
        "\\begin{itemize}\n"
        "  \\item \\textbf{X} ($n=228$): F1-macro \\textbf{0.8386}\n"
        "  \\item \\textbf{Facebook} ($n=232$): F1-macro \\textbf{0.6727}\n"
        "\\end{itemize}\n",
        encoding="utf-8",
    )
    bad_xlmr = (
        "\\begin{itemize}\n"
        "  \\item \\textbf{X} ($n=228$): F1-macro \\textbf{0.8386}\n"
        "\\end{itemize}\n"
    )
    (gen / "platform_prose_xlmr.tex").write_text(bad_xlmr, encoding="utf-8")

    errors = check_platform_metrics(slice_csv, gen)
    assert any("mBERT" in e.message for e in errors)

    good_xlmr = (
        "\\begin{itemize}\n"
        "  \\item \\textbf{X} ($n=228$): F1-macro \\textbf{0.8670}\n"
        "  \\item \\textbf{Facebook} ($n=232$): F1-macro \\textbf{0.7150}\n"
        "\\end{itemize}\n"
    )
    (gen / "platform_prose_xlmr.tex").write_text(good_xlmr, encoding="utf-8")
    assert_platform_consistency(slice_csv, gen)


@pytest.mark.skipif(
    not XLMR_PREDICTIONS_PATH.exists(),
    reason="Predicciones XLM-R auditadas no disponibles",
)
def test_xlmr_audit_platform_values_from_predictions():
    from compute_slice_metrics import compute_all

    slice_df = compute_all(include_xlmr=True)
    xlmr = slice_df[slice_df["model"] == "xlm-roberta"]
    assert not xlmr.empty
    f1_x = xlmr[xlmr["slice"] == "plataforma_X"].iloc[0]["f1_macro"]
    f1_fb = xlmr[xlmr["slice"] == "plataforma_Facebook"].iloc[0]["f1_macro"]
    assert f1_x == pytest.approx(0.867, abs=0.02)
    assert f1_fb == pytest.approx(0.715, abs=0.02)
