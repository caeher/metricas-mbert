"""Métricas por cortes (plataforma, jerga, sarcasmo, ironía)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from discurso_odio.evaluation.metrics import compute_all_metrics

JERGA_DEFINITION = "n_jerga > 0"
DIALECT_BIAS_GROUPS = ("con_jerga", "sin_jerga")


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["SI", "SÍ", "YES", "TRUE", "1"])


def slice_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Calcula métricas por subconjuntos relevantes."""
    results: dict[str, dict[str, Any]] = {}
    work = df.copy()
    work["_y_true"] = y_true
    work["_y_pred"] = y_pred

    if "Plataforma" in work.columns:
        for platform in work["Plataforma"].unique():
            mask = work["Plataforma"] == platform
            idx = mask.values
            prob_slice = y_prob[idx] if y_prob is not None else None
            results[f"plataforma_{platform}"] = compute_all_metrics(
                y_true[idx], y_pred[idx], prob_slice
            )

    if "n_jerga" in work.columns:
        for label, cond in [("con_jerga", work["n_jerga"] > 0), ("sin_jerga", work["n_jerga"] == 0)]:
            idx = cond.values
            prob_slice = y_prob[idx] if y_prob is not None else None
            results[label] = compute_all_metrics(y_true[idx], y_pred[idx], prob_slice)

    for col, name in [
        ("Presencia_Sarcasmo", "sarcasmo"),
        ("Presencia_Ironia", "ironia"),
        ("Contexto_Politico", "contexto_politico"),
    ]:
        if col in work.columns:
            for val, flag in [("si", True), ("no", False)]:
                mask = _to_bool(work[col]) == flag
                idx = mask.values
                if idx.sum() == 0:
                    continue
                prob_slice = y_prob[idx] if y_prob is not None else None
                results[f"{name}_{val}"] = compute_all_metrics(
                    y_true[idx], y_pred[idx], prob_slice
                )

    return results


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
