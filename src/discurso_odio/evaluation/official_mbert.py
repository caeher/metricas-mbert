"""Evaluación oficial y trazabilidad de la métrica mBERT (T-04)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.metrics import bootstrap_ci, compute_all_metrics
from discurso_odio.evaluation.predictions import (
    MBERT_PRED_COL,
    MBERT_PREDICTIONS_JSON,
    MBERT_PREDICTIONS_PATH,
    export_predictions_csv,
    git_commit,
    validate_prediction_vector,
)
from discurso_odio.evaluation.slices import (
    HISTORICAL_PLATFORM_F1,
    export_mbert_slice_artifacts,
    platform_rows,
)
from discurso_odio.models.mbert import default_mbert_model_dir, ensure_checkpoint_contract, predict
from discurso_odio.models.mbert_artifacts import (
    F1_TOLERANCE,
    MANIFEST_REPORT_PATH,
    OFFICIAL_F1_MACRO,
    read_checkpoint_identity,
    sha256_file,
)
from discurso_odio.models.inference_contract import resolve_inference_contract
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS, TEST_CSV

MBERT_PROBABILITIES_PATH = REPORTS_METRICS / "mbert_test_probabilities.csv"
MBERT_METRICS_PATH = REPORTS_METRICS / "mbert_test_metrics.csv"
MBERT_CONFUSION_PATH = REPORTS_METRICS / "mbert_confusion_matrix.csv"
MBERT_CALIBRATION_PATH = REPORTS_METRICS / "mbert_calibration.csv"
MBERT_OFFICIAL_RUN_PATH = REPORTS_METRICS / "mbert_official_run.json"
MBERT_TRACEABILITY_CSV = REPORTS_METRICS / "mbert_f1_traceability.csv"
MBERT_TRACEABILITY_JSON = REPORTS_METRICS / "mbert_f1_traceability.json"
MBERT_PLATFORM_TRACEABILITY_CSV = REPORTS_METRICS / "mbert_platform_traceability.csv"
MBERT_PLATFORM_TRACEABILITY_JSON = REPORTS_METRICS / "mbert_platform_traceability.json"
MODEL_COMPARISON_PATH = REPORTS_METRICS / "model_comparison.csv"

LABEL_NAMES = [
    "No Tóxico",
    "Lenguaje Ofensivo",
    "Discurso de Odio",
    "Amenazas/Violencia",
]


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bins de confiabilidad multiclase (confianza máxima vs acierto)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "confidence_lo": float(lo),
                    "confidence_hi": float(hi),
                    "n": 0,
                    "mean_confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin": i + 1,
                "confidence_lo": float(lo),
                "confidence_hi": float(hi),
                "n": count,
                "mean_confidence": float(confidences[mask].mean()),
                "accuracy": float(accuracies[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def export_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> Path:
    settings = load_settings()
    n_classes = settings["labels"]["num_classes"]
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    rows = []
    for i, true_name in enumerate(LABEL_NAMES):
        for j, pred_name in enumerate(LABEL_NAMES):
            rows.append(
                {
                    "true_label": i,
                    "true_class": true_name,
                    "pred_label": j,
                    "pred_class": pred_name,
                    "count": int(cm[i, j]),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_probabilities_csv(
    test: pd.DataFrame,
    y_prob: np.ndarray,
    path: Path = MBERT_PROBABILITIES_PATH,
) -> Path:
    settings = load_settings()
    n_classes = settings["labels"]["num_classes"]
    data: dict[str, Any] = {"id": test["ID"].values}
    for i in range(n_classes):
        data[f"p{i}"] = y_prob[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_REPORT_PATH.exists():
        with open(MANIFEST_REPORT_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _checkpoint_test_f1(model_dir: Path) -> float | None:
    metrics_path = model_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    value = metrics.get("f1_macro")
    return float(value) if value is not None else None


def _trainer_best_f1(state_path: Path) -> float | None:
    if not state_path.exists():
        return None
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    value = state.get("best_metric")
    return float(value) if value is not None else None


def build_traceability_rows(
    *,
    official_f1_raw: float | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Inventario de cifras F1-macro publicadas o citadas para mBERT."""
    manifest = manifest or _load_manifest()
    settings = load_settings()
    test_sha = sha256_file(TEST_CSV)
    official_raw = official_f1_raw
    if official_raw is None and MBERT_METRICS_PATH.exists():
        official_raw = float(pd.read_csv(MBERT_METRICS_PATH).iloc[0]["f1_macro"])
    if official_raw is None:
        official_raw = OFFICIAL_F1_MACRO

    rows: list[dict[str, Any]] = [
        {
            "f1_macro_published": round(official_raw, 4),
            "rounding": "4dp",
            "source_file": "reports/metrics/mbert_test_metrics.csv",
            "checkpoint": manifest.get("checkpoint", "checkpoint-1072"),
            "git_commit": manifest.get("git_commit") or git_commit(),
            "seed": manifest.get("seed", settings["project"]["seed"]),
            "split": "test",
            "test_sha256": manifest.get("test_sha256", test_sha),
            "evaluated_at": manifest.get("packaged_at"),
            "status": "official",
            "notes": (
                "Checkpoint B2 (mbert-sv-b2): mejor F1-val antes de evaluar test una vez; "
                f"tolerancia {F1_TOLERANCE}"
            ),
        },
        {
            "f1_macro_published": round(official_raw, 3),
            "rounding": "3dp",
            "source_file": "reports/latex/chapters (histórico)",
            "checkpoint": manifest.get("checkpoint", "checkpoint-1072"),
            "git_commit": manifest.get("git_commit") or git_commit(),
            "seed": manifest.get("seed", settings["project"]["seed"]),
            "split": "test",
            "test_sha256": manifest.get("test_sha256", test_sha),
            "evaluated_at": None,
            "status": "same_run_rounded",
            "notes": "Redondeo a 3 decimales del F1 oficial; retirado del informe en T-04",
        },
        {
            "f1_macro_published": 0.7862,
            "rounding": "4dp",
            "source_file": "Informe_Auditoria_de_Modelos.docx (Tabla 7)",
            "checkpoint": "desconocido",
            "git_commit": None,
            "seed": None,
            "split": "test",
            "test_sha256": None,
            "evaluated_at": "2026-08-14",
            "status": "not_reproducible",
            "notes": (
                "Cifra del informe de Cristian citada por la auditoría; "
                "no se reprodujo con checkpoint-1072 ni pesos entregados"
            ),
        },
        {
            "f1_macro_published": 0.7777,
            "rounding": "4dp",
            "source_file": "models/mbert-sv (checkpoint-1072)",
            "checkpoint": "checkpoint-1072",
            "git_commit": None,
            "seed": settings["project"]["seed"],
            "split": "test",
            "test_sha256": test_sha,
            "evaluated_at": "2026-08-16",
            "status": "historical_reproducible",
            "notes": "Checkpoint v1 (train 2,144); sustituido por B2 en sept 2026",
        },
        {
            "f1_macro_published": round(float(manifest.get("best_metric_val_f1_macro", 0.7715)), 4),
            "rounding": "4dp",
            "source_file": "models/mbert-sv-b2/train_metrics.json / val",
            "checkpoint": manifest.get("checkpoint", "checkpoint-1072"),
            "git_commit": manifest.get("git_commit") or git_commit(),
            "seed": manifest.get("seed", settings["project"]["seed"]),
            "split": "val",
            "test_sha256": None,
            "evaluated_at": None,
            "status": "validation_only",
            "notes": "F1-macro de validación (selección de checkpoint); no es métrica de test",
        },
    ]

    # Inspeccionar checkpoints históricos en resultados-metricas
    legacy_root = PROJECT_ROOT / "resultados-metricas" / "models" / "mbert-sv"
    if legacy_root.exists():
        for ckpt_dir in sorted(legacy_root.glob("checkpoint-*")):
            state_path = ckpt_dir / "trainer_state.json"
            best_f1 = _trainer_best_f1(state_path)
            if best_f1 is None:
                continue
            rows.append(
                {
                    "f1_macro_published": round(best_f1, 4),
                    "rounding": "4dp",
                    "source_file": str(state_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "checkpoint": ckpt_dir.name,
                    "git_commit": None,
                    "seed": settings["project"]["seed"],
                    "split": "val",
                    "test_sha256": None,
                    "evaluated_at": None,
                    "status": "validation_only",
                    "notes": "Checkpoint intermedio de entrenamiento; no oficial",
                }
            )
        legacy_test_f1 = _checkpoint_test_f1(legacy_root)
        if legacy_test_f1 is not None and abs(legacy_test_f1 - official_raw) > F1_TOLERANCE:
            rows.append(
                {
                    "f1_macro_published": round(legacy_test_f1, 4),
                    "rounding": "4dp",
                    "source_file": "resultados-metricas/models/mbert-sv/test_metrics.json",
                    "checkpoint": "mbert-sv (legacy)",
                    "git_commit": None,
                    "seed": settings["project"]["seed"],
                    "split": "test",
                    "test_sha256": test_sha,
                    "evaluated_at": None,
                    "status": "not_reproducible",
                    "notes": "Test F1 de copia legacy; distinto del checkpoint oficial T-02",
                }
            )

    return rows


def export_traceability(
    *,
    official_f1_raw: float | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    rows = build_traceability_rows(official_f1_raw=official_f1_raw, manifest=manifest)
    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(MBERT_TRACEABILITY_CSV, index=False)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_status": "official",
        "official_f1_macro": round(
            official_f1_raw if official_f1_raw is not None else OFFICIAL_F1_MACRO, 4
        ),
        "rows": rows,
    }
    with open(MBERT_TRACEABILITY_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return MBERT_TRACEABILITY_CSV, MBERT_TRACEABILITY_JSON


def _refresh_model_comparison(mbert_f1: float) -> None:
    if not MODEL_COMPARISON_PATH.exists():
        return
    df = pd.read_csv(MODEL_COMPARISON_PATH)
    if "model" not in df.columns or "f1_macro" not in df.columns:
        return
    mask = df["model"].astype(str).str.lower() == "mbert"
    if mask.any():
        df.loc[mask, "f1_macro"] = mbert_f1
        df.to_csv(MODEL_COMPARISON_PATH, index=False)


def load_official_run_metadata() -> dict[str, Any] | None:
    """Lee metadatos de la corrida oficial T-04 si existen."""
    if not MBERT_OFFICIAL_RUN_PATH.exists():
        return None
    with open(MBERT_OFFICIAL_RUN_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def official_checkpoint_label(
    official: dict[str, Any] | None = None,
    model_dir: Path | None = None,
) -> str:
    """Identificador corto de checkpoint para artefactos (no ruta absoluta)."""
    official = official or load_official_run_metadata()
    if official and official.get("checkpoint"):
        return str(official["checkpoint"])
    model_dir = model_dir or default_mbert_model_dir()
    identity = read_checkpoint_identity(model_dir)
    if identity.get("checkpoint"):
        return str(identity["checkpoint"])
    return "checkpoint-1072"


def build_platform_traceability_rows(
    *,
    slice_df: pd.DataFrame | None = None,
    official: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Inventario de F1 por plataforma: oficial vs cifras históricas no reproducibles."""
    official = official or load_official_run_metadata()
    settings = load_settings()
    checkpoint = official_checkpoint_label(official)
    commit = (official or {}).get("git_commit") or git_commit()
    seed = (official or {}).get("seed", settings["project"]["seed"])
    test_sha = (official or {}).get("test_sha256") or sha256_file(TEST_CSV)
    evaluated_at = (official or {}).get("evaluated_at")

    rows: list[dict[str, Any]] = []

    if slice_df is None and (REPORTS_METRICS / "slice_metrics.csv").exists():
        slice_df = pd.read_csv(REPORTS_METRICS / "slice_metrics.csv")

    if slice_df is not None and not slice_df.empty:
        mbert_platform = platform_rows(slice_df[slice_df["model"] == "mbert"])
        for _, row in mbert_platform.iterrows():
            rows.append(
                {
                    "platform": str(row["slice"]).replace("plataforma_", ""),
                    "f1_macro_published": round(float(row["f1_macro"]), 4),
                    "n": int(row["n"]),
                    "accuracy": round(float(row["accuracy"]), 4),
                    "source_file": "reports/metrics/slice_metrics.csv",
                    "checkpoint": checkpoint,
                    "git_commit": commit,
                    "seed": seed,
                    "split": "test",
                    "test_sha256": test_sha,
                    "evaluated_at": evaluated_at,
                    "status": "official",
                    "notes": (
                        "Corte por plataforma de la evaluación oficial T-04/T-05; "
                        f"run_id={(official or {}).get('run_id')}"
                    ),
                }
            )

    for slice_name, historical_f1 in HISTORICAL_PLATFORM_F1.items():
        platform = slice_name.replace("plataforma_", "")
        rows.append(
            {
                "platform": platform,
                "f1_macro_published": round(historical_f1, 4),
                "n": None,
                "accuracy": None,
                "source_file": "Informe de Cristian / auditoría (Tabla plataforma)",
                "checkpoint": "desconocido",
                "git_commit": None,
                "seed": None,
                "split": "test",
                "test_sha256": None,
                "evaluated_at": "2026-08-14",
                "status": "not_reproducible",
                "notes": (
                    "Cifra histórica citada por la auditoría; no reproducible con "
                    f"checkpoint oficial {checkpoint}"
                ),
            }
        )

    return rows


def export_platform_traceability(
    *,
    slice_df: pd.DataFrame | None = None,
    official: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    rows = build_platform_traceability_rows(slice_df=slice_df, official=official)
    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(MBERT_PLATFORM_TRACEABILITY_CSV, index=False)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    with open(MBERT_PLATFORM_TRACEABILITY_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return MBERT_PLATFORM_TRACEABILITY_CSV, MBERT_PLATFORM_TRACEABILITY_JSON


def evaluate_mbert_official(
    *,
    model_dir: Path | None = None,
    max_length: int | None = None,
    n_bootstrap: int = 1000,
    skip_official_f1_check: bool = False,
) -> dict[str, Any]:
    """Evalúa el checkpoint oficial una vez sobre test y exporta artefactos alineados."""
    model_dir = model_dir or default_mbert_model_dir()
    model_dir = Path(model_dir)
    ensure_checkpoint_contract(model_dir)
    contract = resolve_inference_contract(model_dir)
    max_length = contract.validate_max_length(max_length)
    manifest = _load_manifest()
    identity = read_checkpoint_identity(model_dir)
    settings = load_settings()

    test = load_split("test")
    text_col = contract.text_column
    y_pred, y_prob = predict(
        test[text_col].tolist(),
        model_dir,
        max_length=max_length,
        normalize=True,
    )
    y_true = test["label"].to_numpy()

    pred_df = pd.DataFrame({"id": test["ID"].values, MBERT_PRED_COL: y_pred})
    validate_prediction_vector(test, pred_df, MBERT_PRED_COL, "mBERT")

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(
        y_true,
        y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        n_bootstrap=n_bootstrap,
        seed=settings["project"]["seed"],
    )
    metrics["f1_bootstrap_mean"] = f1_mean
    metrics["f1_bootstrap_lo"] = f1_lo
    metrics["f1_bootstrap_hi"] = f1_hi

    evaluated_at = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid.uuid4())
    commit = git_commit()
    test_sha = sha256_file(TEST_CSV)
    f1_raw = float(metrics["f1_macro"])
    f1_official = round(f1_raw, 4)
    checkpoint_label = (
        identity.get("checkpoint") or manifest.get("checkpoint", "checkpoint-1072")
    )

    run_metadata = {
        "run_id": run_id,
        "evaluated_at": evaluated_at,
        "git_commit": commit,
        "seed": settings["project"]["seed"],
        "base_model": settings["model"]["name"],
        "checkpoint": checkpoint_label,
        "best_model_checkpoint": identity.get("best_model_checkpoint"),
        "best_metric_val_f1_macro": identity.get("best_metric_val_f1_macro"),
        "model_dir": str(model_dir.resolve()),
        "split": TEST_CSV.name,
        "test_csv": str(TEST_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_n": len(test),
        "test_sha256": test_sha,
        "max_length": max_length,
        "text_column": contract.text_column,
        "normalizer": contract.normalizer,
        "normalizer_version": contract.normalizer_version,
        "f1_macro_raw": f1_raw,
        "official_f1_macro": f1_official,
        "f1_tolerance": F1_TOLERANCE,
        "f1_bootstrap_lo": f1_lo,
        "f1_bootstrap_hi": f1_hi,
        "artifacts": {
            "predictions_csv": str(MBERT_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "probabilities_csv": str(MBERT_PROBABILITIES_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "metrics_csv": str(MBERT_METRICS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "confusion_csv": str(MBERT_CONFUSION_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "calibration_csv": str(MBERT_CALIBRATION_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "traceability_csv": str(MBERT_TRACEABILITY_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "slice_metrics_csv": "reports/metrics/slice_metrics.csv",
            "slice_perclass_csv": "reports/metrics/slice_perclass_metrics.csv",
            "dialect_bias_csv": "reports/metrics/dialect_bias_probe.csv",
            "platform_traceability_csv": "reports/metrics/mbert_platform_traceability.csv",
        },
    }

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)

    export_predictions_csv(
        test,
        y_pred,
        csv_path=MBERT_PREDICTIONS_PATH,
        json_path=MBERT_PREDICTIONS_JSON,
        pred_col=MBERT_PRED_COL,
        model="mbert",
        checkpoint=checkpoint_label,
        max_length=max_length,
        split=TEST_CSV.name,
        extra_metadata={
            "run_id": run_id,
            "evaluated_at": evaluated_at,
            "official_f1_macro": f1_official,
            **contract.metadata(),
        },
    )
    export_probabilities_csv(test, y_prob, MBERT_PROBABILITIES_PATH)

    slice_artifacts = export_mbert_slice_artifacts(
        test,
        y_true,
        y_pred,
        y_prob,
        REPORTS_METRICS,
        run_id=run_id,
        checkpoint=checkpoint_label,
        split=TEST_CSV.name,
        extra_metadata={
            "evaluated_at": evaluated_at,
            "git_commit": commit,
            "official_f1_macro": f1_official,
        },
        merge=True,
    )
    slice_df = pd.read_csv(slice_artifacts["slice_metrics_csv"])
    export_platform_traceability(slice_df=slice_df, official=run_metadata)

    export_row = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
    pd.DataFrame([export_row]).to_csv(MBERT_METRICS_PATH, index=False)

    export_confusion_matrix(y_true, y_pred, MBERT_CONFUSION_PATH)
    calibration_bins(y_true, y_prob).to_csv(MBERT_CALIBRATION_PATH, index=False)

    with open(MBERT_OFFICIAL_RUN_PATH, "w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with open(model_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(export_row, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    _refresh_model_comparison(f1_raw)
    export_traceability(official_f1_raw=f1_raw, manifest=manifest)

    if not skip_official_f1_check and abs(f1_official - OFFICIAL_F1_MACRO) > F1_TOLERANCE:
        raise ValueError(
            f"F1 oficial {f1_official} fuera de tolerancia respecto a {OFFICIAL_F1_MACRO}"
        )

    return run_metadata
