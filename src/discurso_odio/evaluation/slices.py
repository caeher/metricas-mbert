"""Métricas por cortes (plataforma, jerga, sarcasmo, ironía, contexto político)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from discurso_odio.evaluation.indicators import normalize_indicator, resolve_slice_columns
from discurso_odio.evaluation.metrics import bootstrap_ci, compute_all_metrics

JERGA_DEFINITION = "n_jerga > 0"
DIALECT_BIAS_GROUPS = ("con_jerga", "sin_jerga")
MIN_SLICE_N_FOR_CI = 30

INDICATOR_SLICE_DEFS: tuple[tuple[str, str], ...] = (
    ("Presencia_Sarcasmo", "sarcasmo"),
    ("Presencia_Ironia", "ironia"),
    ("Contexto_Politico", "contexto_politico"),
)


def _empty_slice_metrics() -> dict[str, Any]:
    return {
        "n": 0,
        "f1_macro": float("nan"),
        "accuracy": float("nan"),
        "f1_ci_lo": float("nan"),
        "f1_ci_hi": float("nan"),
    }


def _compute_slice_entry(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    mask: np.ndarray,
) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return _empty_slice_metrics()

    yt = y_true[mask]
    yp = y_pred[mask]
    prob_slice = y_prob[mask] if y_prob is not None else None
    metrics = compute_all_metrics(yt, yp, prob_slice)

    f1_ci_lo = float("nan")
    f1_ci_hi = float("nan")
    if n >= MIN_SLICE_N_FOR_CI and len(np.unique(yt)) >= 2:
        _, f1_ci_lo, f1_ci_hi = bootstrap_ci(
            yt,
            yp,
            lambda y_t, y_p: f1_score(y_t, y_p, average="macro", zero_division=0),
        )

    return {
        "n": n,
        "f1_macro": float(metrics["f1_macro"]),
        "accuracy": float(metrics["accuracy"]),
        "f1_ci_lo": f1_ci_lo,
        "f1_ci_hi": f1_ci_hi,
    }


def slice_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    include_empty_slices: bool = True,
) -> dict[str, dict[str, Any]]:
    """Calcula métricas por subconjuntos relevantes."""
    results: dict[str, dict[str, Any]] = {}
    work = df.copy()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if "Plataforma" in work.columns:
        for platform in sorted(work["Plataforma"].dropna().unique(), key=str):
            mask = (work["Plataforma"] == platform).values
            entry = _compute_slice_entry(y_true, y_pred, y_prob, mask)
            if include_empty_slices or entry["n"] > 0:
                results[f"plataforma_{platform}"] = entry

    if "n_jerga" in work.columns:
        for label, cond in [("con_jerga", work["n_jerga"] > 0), ("sin_jerga", work["n_jerga"] == 0)]:
            mask = cond.values
            entry = _compute_slice_entry(y_true, y_pred, y_prob, mask)
            if include_empty_slices or entry["n"] > 0:
                results[label] = entry

    resolved = resolve_slice_columns(work)
    for canonical, slice_name in INDICATOR_SLICE_DEFS:
        col = resolved.get(canonical)
        if col is None:
            continue
        normalized = normalize_indicator(work[col])
        for suffix, flag in [("si", True), ("no", False)]:
            mask = (normalized == flag).fillna(False).values
            entry = _compute_slice_entry(y_true, y_pred, y_prob, mask)
            if include_empty_slices or entry["n"] > 0:
                results[f"{slice_name}_{suffix}"] = entry

    return results


def slice_metrics_to_rows(
    slices: dict[str, dict[str, Any]],
    *,
    model: str,
    checkpoint: str,
    split: str,
    run_id: str | None = None,
    n_test: int | None = None,
) -> list[dict[str, Any]]:
    """Convierte el dict de slice_metrics en filas tabulares exportables."""
    run_id = run_id or str(uuid4())
    rows: list[dict[str, Any]] = []
    for slice_name, metrics in sorted(slices.items()):
        if metrics.get("n", 0) == 0:
            continue
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "checkpoint": checkpoint,
                "split": split,
                "n_test": n_test,
                "slice": slice_name,
                "n": metrics["n"],
                "f1_macro": metrics["f1_macro"],
                "accuracy": metrics["accuracy"],
                "f1_ci_lo": metrics.get("f1_ci_lo", float("nan")),
                "f1_ci_hi": metrics.get("f1_ci_hi", float("nan")),
            }
        )
    return rows


def _wilson_ci(fp: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confianza Wilson 95 % para una proporción binomial."""
    from scipy import stats

    if n <= 0:
        return float("nan"), float("nan")
    result = stats.binomtest(fp, n, p=0.5)
    lo, hi = result.proportion_ci(confidence_level=1 - alpha, method="wilson")
    return float(lo), float(hi)


def dialect_bias_probe(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model: str = "unknown",
    checkpoint: str = "",
    run_id: str | None = None,
    split: str = "test.csv",
    jerga_definition: str = JERGA_DEFINITION,
) -> pd.DataFrame:
    """Tasa de FP en No Tóxico con y sin jerga local.

    Definición: clase real No Tóxico (y_true == 0); FP si y_pred >= 1;
    grupos según n_jerga > 0 vs n_jerga == 0.
    """
    if "n_jerga" not in df.columns:
        return pd.DataFrame(
            columns=[
                "run_id",
                "model",
                "checkpoint",
                "split",
                "n_test",
                "group",
                "n",
                "fp",
                "fpr",
                "fpr_ci_lo",
                "fpr_ci_hi",
                "jerga_definition",
            ]
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    not_toxic = y_true == 0
    fp_mask = y_pred >= 1
    run_id = run_id or str(uuid4())

    group_masks = {
        "con_jerga": (df["n_jerga"] > 0).values & not_toxic,
        "sin_jerga": (df["n_jerga"] == 0).values & not_toxic,
    }

    rows: list[dict[str, Any]] = []
    for group, mask in group_masks.items():
        n = int(mask.sum())
        fp = int((fp_mask & mask).sum()) if n > 0 else 0
        if n > 0:
            fpr = fp / n
            ci_lo, ci_hi = _wilson_ci(fp, n)
        else:
            fpr = float("nan")
            ci_lo, ci_hi = float("nan"), float("nan")

        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "checkpoint": checkpoint,
                "split": split,
                "n_test": len(df),
                "group": group,
                "n": n,
                "fp": fp,
                "fpr": fpr,
                "fpr_ci_lo": ci_lo,
                "fpr_ci_hi": ci_hi,
                "jerga_definition": jerga_definition,
            }
        )

    return pd.DataFrame(rows)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def export_dialect_bias(
    probe_df: pd.DataFrame,
    output_dir: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Exporta la sonda dialectal a CSV y JSON con metadatos de ejecución."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dialect_bias_probe.csv"
    json_path = output_dir / "dialect_bias_probe.json"

    probe_df.to_csv(csv_path, index=False)

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "jerga_definition": JERGA_DEFINITION,
        "rows": probe_df.to_dict(orient="records"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


PLATFORM_SLICES = ("plataforma_X", "plataforma_Facebook")


def platform_rows(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Filtra filas de cortes por plataforma (X y Facebook)."""
    if slice_df.empty or "slice" not in slice_df.columns:
        return slice_df.iloc[0:0].copy()
    return slice_df[slice_df["slice"].isin(PLATFORM_SLICES)].copy()


def _merge_slice_metrics(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Fusiona por (model, slice) preservando filas de otros modelos."""
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()
    if "model" not in incoming.columns or "slice" not in incoming.columns:
        return incoming.copy()

    models_incoming = set(incoming["model"].dropna().unique())
    if "model" in existing.columns:
        keep = existing[~existing["model"].isin(models_incoming)]
        merged = pd.concat([keep, incoming], ignore_index=True)
    else:
        merged = incoming.copy()

    sort_cols = [c for c in ("model", "slice") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged


def export_slice_metrics(
    slice_df: pd.DataFrame,
    output_dir: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
    merge: bool = True,
) -> tuple[Path, Path]:
    """Exporta métricas por cortes a CSV y JSON con metadatos de ejecución."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "slice_metrics.csv"
    json_path = output_dir / "slice_metrics.json"

    if merge and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
            slice_df = _merge_slice_metrics(existing, slice_df)
        except (OSError, pd.errors.EmptyDataError, ValueError):
            pass

    slice_df.to_csv(csv_path, index=False)

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "min_n_for_ci": MIN_SLICE_N_FOR_CI,
        "rows": slice_df.to_dict(orient="records"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return csv_path, json_path
