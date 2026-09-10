"""Análisis de errores mBERT sobre test oficial (Fase 0 del plan F1 0.80)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split, summarize_train_platform_class_weights
from discurso_odio.evaluation.predictions import MBERT_PRED_COL, MBERT_PREDICTIONS_PATH
from discurso_odio.paths import REPORTS_METRICS

ERROR_ANALYSIS_JSON = REPORTS_METRICS / "error_analysis_b2.json"
ERROR_ANALYSIS_CSV = REPORTS_METRICS / "error_analysis_b2_errors.csv"

LABEL_NAMES = load_settings()["labels"]["names"]


def _is_adjacent(true_label: int, pred_label: int) -> bool:
    return abs(int(true_label) - int(pred_label)) == 1


def _row_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "id": int(row["ID"]) if "ID" in row else None,
        "platform": str(row.get("Plataforma", "")),
        "true_label": int(row["label"]),
        "true_class": LABEL_NAMES[int(row["label"])],
        "pred_label": int(row["y_pred"]),
        "pred_class": LABEL_NAMES[int(row["y_pred"])],
        "adjacent_error": _is_adjacent(row["label"], row["y_pred"]),
        "n_chars": int(row.get("n_chars", 0) or 0),
        "n_jerga": int(row.get("n_jerga", 0) or 0),
        "ironia": str(row.get("Presencia_Ironia", "")),
        "sarcasmo": str(row.get("Presencia_Sarcasmo", "")),
        "contexto_politico": str(row.get("Contexto_Politico", "")),
    }


def merge_test_predictions(
    test_df: pd.DataFrame | None = None,
    predictions_path: Path | None = None,
    pred_col: str = MBERT_PRED_COL,
) -> pd.DataFrame:
    """Une test.csv con predicciones oficiales por ID."""
    test_df = test_df if test_df is not None else load_split("test")
    pred_path = predictions_path or MBERT_PREDICTIONS_PATH
    if not pred_path.exists():
        raise FileNotFoundError(f"Predicciones no encontradas: {pred_path}")

    preds = pd.read_csv(pred_path)
    id_col = "id" if "id" in preds.columns else preds.columns[0]
    pred_name = pred_col if pred_col in preds.columns else preds.columns[1]

    merged = test_df.merge(
        preds.rename(columns={id_col: "ID", pred_name: "y_pred"}),
        on="ID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(test_df):
        raise ValueError(
            f"Merge incompleto: test={len(test_df)} merged={len(merged)} preds={len(preds)}"
        )
    return merged


def analyze_errors(
    merged: pd.DataFrame,
    *,
    model: str = "mbert-b2",
    checkpoint: str = "checkpoint-1644",
) -> dict[str, Any]:
    """Genera resumen estructurado de errores y pesos de train."""
    y_true = merged["label"].astype(int).to_numpy()
    y_pred = merged["y_pred"].astype(int).to_numpy()
    errors_mask = y_true != y_pred
    errors_df = merged.loc[errors_mask].copy()

    adjacent = sum(_is_adjacent(t, p) for t, p in zip(y_true[errors_mask], y_pred[errors_mask]))
    n_errors = int(errors_mask.sum())

    by_platform: dict[str, dict[str, Any]] = {}
    for platform, group in merged.groupby("Plataforma"):
        yt = group["label"].astype(int).to_numpy()
        yp = group["y_pred"].astype(int).to_numpy()
        err = group.loc[group["label"] != group["y_pred"]]
        by_platform[str(platform)] = {
            "n": int(len(group)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "n_errors": int(len(err)),
            "error_rate": float(len(err) / len(group)) if len(group) else 0.0,
        }

    by_confusion: list[dict[str, Any]] = []
    confusion_counts = Counter(
        (int(r["label"]), int(r["y_pred"])) for _, r in errors_df.iterrows()
    )
    for (true_l, pred_l), count in sorted(confusion_counts.items(), key=lambda x: -x[1]):
        by_confusion.append(
            {
                "true_label": true_l,
                "true_class": LABEL_NAMES[true_l],
                "pred_label": pred_l,
                "pred_class": LABEL_NAMES[pred_l],
                "count": count,
                "adjacent": _is_adjacent(true_l, pred_l),
            }
        )

    facebook_odio_mask = (merged["Plataforma"] == "Facebook") & (merged["label"] == 2)
    facebook_odio_support = int(facebook_odio_mask.sum())
    facebook_odio_errors = merged.loc[facebook_odio_mask & errors_mask]
    facebook_odio_hits = merged.loc[facebook_odio_mask & ~errors_mask]

    priority_slices: dict[str, Any] = {
        "facebook_odio_errors": int(len(facebook_odio_errors)),
        "facebook_odio_support": facebook_odio_support,
        "facebook_odio_recall": float(
            len(facebook_odio_hits) / max(1, facebook_odio_support)
        ),
        "ironia_si_f1": float(
            f1_score(
                merged.loc[merged["Presencia_Ironia"].astype(str).str.upper().isin({"SI", "SÍ"})]["label"],
                merged.loc[merged["Presencia_Ironia"].astype(str).str.upper().isin({"SI", "SÍ"})]["y_pred"],
                average="macro",
                zero_division=0,
            )
            if (merged["Presencia_Ironia"].astype(str).str.upper().isin({"SI", "SÍ"})).any()
            else None
        ),
    }

    error_rows = [_row_metadata(row) for _, row in errors_df.iterrows()]

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "checkpoint": checkpoint,
        "n_test": int(len(merged)),
        "n_errors": n_errors,
        "error_rate": float(n_errors / len(merged)),
        "adjacent_error_count": adjacent,
        "adjacent_error_pct": float(adjacent / n_errors * 100) if n_errors else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "by_platform": by_platform,
        "by_confusion": by_confusion,
        "priority_slices": priority_slices,
        "train_platform_class_weights": summarize_train_platform_class_weights(),
        "recommendations": _build_recommendations(by_platform, by_confusion, priority_slices),
        "errors": error_rows,
    }


def _build_recommendations(
    by_platform: dict[str, dict[str, Any]],
    by_confusion: list[dict[str, Any]],
    priority_slices: dict[str, Any],
) -> list[str]:
    recs: list[str] = []
    fb = by_platform.get("Facebook", {})
    if fb.get("f1_macro", 1.0) < 0.75:
        recs.append(
            "Subir peso de muestra en train para Facebook+Odio y Facebook+Amenazas; "
            "opcionalmente bajar Facebook+Ofensivo."
        )
    top = by_confusion[:3] if by_confusion else []
    if top and top[0].get("adjacent"):
        recs.append(
            "Confusiones adyacentes dominan: probar pérdida ordinal o focal en clases 1-3."
        )
    if priority_slices.get("facebook_odio_recall", 1.0) < 0.5:
        recs.append(
            "Recall bajo en odio/Facebook: ablar sintéticos emoji/relleno o downweight global."
        )
    recs.append("Quitar o reducir label_smoothing=0.05 si se busca recuperar F1 puntual v1.")
    return recs


def export_error_analysis(
    output_json: Path | None = None,
    output_csv: Path | None = None,
    *,
    predictions_path: Path | None = None,
    model: str = "mbert-b2",
    checkpoint: str = "checkpoint-1644",
) -> dict[str, Any]:
    """Ejecuta análisis y escribe artefactos."""
    merged = merge_test_predictions(predictions_path=predictions_path)
    report = analyze_errors(merged, model=model, checkpoint=checkpoint)

    json_path = output_json or ERROR_ANALYSIS_JSON
    csv_path = output_csv or ERROR_ANALYSIS_CSV
    json_path.parent.mkdir(parents=True, exist_ok=True)

    export_payload = {k: v for k, v in report.items() if k != "errors"}
    export_payload["n_error_rows_exported"] = len(report["errors"])
    json_path.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame(report["errors"]).to_csv(csv_path, index=False)
    return report
