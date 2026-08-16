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
from sklearn.metrics import f1_score, precision_recall_fscore_support

from discurso_odio.config import load_settings
from discurso_odio.evaluation.indicators import normalize_indicator, resolve_slice_columns
from discurso_odio.evaluation.metrics import bootstrap_ci, compute_all_metrics

JERGA_DEFINITION = "n_jerga > 0"
DIALECT_BIAS_GROUPS = ("con_jerga", "sin_jerga")
MIN_SLICE_N_FOR_CI = 30
MIN_SLICE_CLASS_SUPPORT = 5
SLICE_PERCLASS_METRICS_PATH_NAME = "slice_perclass_metrics.csv"
SLICE_METRICS_PATH_NAME = "slice_metrics.csv"
SLICE_METRICS_JSON_NAME = "slice_metrics.json"

INDICATOR_SLICE_DEFS: tuple[tuple[str, str], ...] = (
    ("Presencia_Sarcasmo", "sarcasmo"),
    ("Presencia_Ironia", "ironia"),
    ("Contexto_Politico", "contexto_politico"),
)

LABEL_NAMES = [
    "No Tóxico",
    "Lenguaje Ofensivo",
    "Discurso de Odio",
    "Amenazas/Violencia",
]


def _num_classes() -> int:
    return int(load_settings()["labels"]["num_classes"])


def _empty_slice_metrics() -> dict[str, Any]:
    n_classes = _num_classes()
    entry: dict[str, Any] = {
        "n": 0,
        "definition": "",
        "f1_macro": float("nan"),
        "accuracy": float("nan"),
        "f1_ci_lo": float("nan"),
        "f1_ci_hi": float("nan"),
        "low_power": True,
        "power_note": "sin observaciones",
    }
    for i in range(n_classes):
        entry[f"n{i}"] = 0
    return entry


def _class_counts(y_true: np.ndarray, n_classes: int) -> dict[str, int]:
    counts = {f"n{i}": 0 for i in range(n_classes)}
    for label in range(n_classes):
        counts[f"n{label}"] = int(np.sum(y_true == label))
    return counts


def _assess_low_power(n: int, counts: dict[str, int], n_classes: int) -> tuple[bool, str]:
    if n == 0:
        return True, "sin observaciones"
    supports = [counts[f"n{i}"] for i in range(n_classes)]
    present = sum(1 for s in supports if s > 0)
    if n < MIN_SLICE_N_FOR_CI:
        return True, f"baja potencia (n={n})"
    if present < n_classes:
        return True, f"baja potencia: faltan clases (n={n})"
    if any(s < MIN_SLICE_CLASS_SUPPORT for s in supports if s > 0):
        return True, f"baja potencia: clase con n<{MIN_SLICE_CLASS_SUPPORT}"
    return False, ""


def _power_note(
    n: int,
    low_power: bool,
    note: str,
    f1_ci_lo: float,
    f1_ci_hi: float,
) -> str:
    if low_power:
        return note or f"baja potencia (n={n})"
    if pd.notna(f1_ci_lo) and pd.notna(f1_ci_hi):
        return f"IC 95% [{f1_ci_lo:.4f}, {f1_ci_hi:.4f}]"
    return f"n={n}"


def _per_class_rows(
    slice_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model: str,
    checkpoint: str,
    run_id: str,
    split: str,
    n_test: int | None,
) -> list[dict[str, Any]]:
    n_classes = _num_classes()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(LABEL_NAMES[:n_classes]):
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "checkpoint": checkpoint,
                "split": split,
                "n_test": n_test,
                "slice": slice_name,
                "class_label": i,
                "class_name": name,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        )
    return rows


def _compute_slice_entry(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    mask: np.ndarray,
    *,
    definition: str = "",
) -> dict[str, Any]:
    n_classes = _num_classes()
    n = int(mask.sum())
    if n == 0:
        entry = _empty_slice_metrics()
        entry["definition"] = definition
        return entry

    yt = y_true[mask]
    yp = y_pred[mask]
    prob_slice = y_prob[mask] if y_prob is not None else None
    metrics = compute_all_metrics(yt, yp, prob_slice)

    counts = _class_counts(yt, n_classes)
    low_power, low_note = _assess_low_power(n, counts, n_classes)

    f1_ci_lo = float("nan")
    f1_ci_hi = float("nan")
    if n >= MIN_SLICE_N_FOR_CI and len(np.unique(yt)) >= 2 and not low_power:
        _, f1_ci_lo, f1_ci_hi = bootstrap_ci(
            yt,
            yp,
            lambda y_t, y_p: f1_score(y_t, y_p, average="macro", zero_division=0),
        )

    entry: dict[str, Any] = {
        "n": n,
        "definition": definition,
        "f1_macro": float(metrics["f1_macro"]),
        "accuracy": float(metrics["accuracy"]),
        "f1_ci_lo": f1_ci_lo,
        "f1_ci_hi": f1_ci_hi,
        "low_power": low_power,
        "power_note": _power_note(n, low_power, low_note, f1_ci_lo, f1_ci_hi),
        **counts,
    }

    if n >= MIN_SLICE_N_FOR_CI:
        entry["per_class"] = _per_class_rows(
            "",
            yt,
            yp,
            model="",
            checkpoint="",
            run_id="",
            split="",
            n_test=None,
        )

    return entry


def _indicator_definition(canonical: str, flag: bool) -> str:
    suffix = "SI/SÍ/YES/TRUE/1" if flag else "NO/FALSE/0/N"
    aliases = {
        "Presencia_Sarcasmo": "Presencia_Sarcasmo",
        "Presencia_Ironia": "Presencia_Ironia",
        "Contexto_Politico": "Contexto_Politico",
    }
    col = aliases.get(canonical, canonical)
    return f"{col} in {{{suffix}}}"


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
            definition = f"Plataforma == '{platform}'"
            entry = _compute_slice_entry(
                y_true, y_pred, y_prob, mask, definition=definition
            )
            if include_empty_slices or entry["n"] > 0:
                results[f"plataforma_{platform}"] = entry

    if "n_jerga" in work.columns:
        for label, cond, definition in [
            ("con_jerga", work["n_jerga"] > 0, JERGA_DEFINITION),
            ("sin_jerga", work["n_jerga"] == 0, "n_jerga == 0"),
        ]:
            mask = cond.values
            entry = _compute_slice_entry(
                y_true, y_pred, y_prob, mask, definition=definition
            )
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
            definition = _indicator_definition(canonical, flag)
            entry = _compute_slice_entry(
                y_true, y_pred, y_prob, mask, definition=definition
            )
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
    n_classes = _num_classes()
    rows: list[dict[str, Any]] = []
    for slice_name, metrics in sorted(slices.items()):
        if metrics.get("n", 0) == 0:
            continue
        row: dict[str, Any] = {
            "run_id": run_id,
            "model": model,
            "checkpoint": checkpoint,
            "split": split,
            "n_test": n_test,
            "slice": slice_name,
            "definition": metrics.get("definition", ""),
            "n": metrics["n"],
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "f1_ci_lo": metrics.get("f1_ci_lo", float("nan")),
            "f1_ci_hi": metrics.get("f1_ci_hi", float("nan")),
            "low_power": bool(metrics.get("low_power", False)),
            "power_note": metrics.get("power_note", ""),
        }
        for i in range(n_classes):
            row[f"n{i}"] = metrics.get(f"n{i}", 0)
        rows.append(row)
    return rows


def collect_perclass_rows(
    slices: dict[str, dict[str, Any]],
    *,
    model: str,
    checkpoint: str,
    split: str,
    run_id: str,
    n_test: int | None,
) -> list[dict[str, Any]]:
    """Recoge métricas por clase solo para cortes con n >= MIN_SLICE_N_FOR_CI."""
    all_rows: list[dict[str, Any]] = []
    for slice_name, metrics in sorted(slices.items()):
        n = metrics.get("n", 0)
        if n < MIN_SLICE_N_FOR_CI or "per_class" not in metrics:
            continue
        for row in metrics["per_class"]:
            row = dict(row)
            row["slice"] = slice_name
            row["model"] = model
            row["checkpoint"] = checkpoint
            row["split"] = split
            row["run_id"] = run_id
            row["n_test"] = n_test
            all_rows.append(row)
    return all_rows


def format_class_distribution(row: dict[str, Any] | pd.Series, n_classes: int | None = None) -> str:
    """Formato n0/n1/n2/n3 para tablas del informe."""
    n_classes = n_classes or _num_classes()
    parts = [str(int(row.get(f"n{i}", 0))) for i in range(n_classes)]
    return "/".join(parts)


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


def _merge_by_model(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Fusiona por modelo preservando filas de otros modelos."""
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()
    if "model" not in incoming.columns:
        return incoming.copy()

    models_incoming = set(incoming["model"].dropna().unique())
    if "model" in existing.columns:
        keep = existing[~existing["model"].isin(models_incoming)]
        merged = pd.concat([keep, incoming], ignore_index=True)
    else:
        merged = incoming.copy()

    sort_cols = [c for c in ("model", "slice", "group") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged


def export_dialect_bias(
    probe_df: pd.DataFrame,
    output_dir: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
    merge: bool = True,
) -> tuple[Path, Path]:
    """Exporta la sonda dialectal a CSV y JSON con metadatos de ejecución."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dialect_bias_probe.csv"
    json_path = output_dir / "dialect_bias_probe.json"

    if merge and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
            probe_df = _merge_by_model(existing, probe_df)
        except (OSError, pd.errors.EmptyDataError, ValueError):
            pass

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

HISTORICAL_PLATFORM_F1 = {
    "plataforma_X": 0.8503,
    "plataforma_Facebook": 0.6896,
}


def platform_rows(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Filtra filas de cortes por plataforma (X y Facebook)."""
    if slice_df.empty or "slice" not in slice_df.columns:
        return slice_df.iloc[0:0].copy()
    return slice_df[slice_df["slice"].isin(PLATFORM_SLICES)].copy()


def _merge_slice_metrics(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Fusiona por (model, slice) preservando filas de otros modelos."""
    return _merge_by_model(existing, incoming)


def export_slice_perclass_metrics(
    perclass_df: pd.DataFrame,
    output_dir: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
    merge: bool = True,
) -> Path:
    """Exporta métricas por clase en cortes con n suficiente."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / SLICE_PERCLASS_METRICS_PATH_NAME

    if merge and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
            perclass_df = _merge_by_model(existing, perclass_df)
        except (OSError, pd.errors.EmptyDataError, ValueError):
            pass

    perclass_df.to_csv(csv_path, index=False)

    if extra_metadata:
        json_path = output_dir / "slice_perclass_metrics.json"
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "rows": perclass_df.to_dict(orient="records"),
            **extra_metadata,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return csv_path


def export_slice_metrics(
    slice_df: pd.DataFrame,
    output_dir: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
    merge: bool = True,
) -> tuple[Path, Path]:
    """Exporta métricas por cortes a CSV y JSON con metadatos de ejecución."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / SLICE_METRICS_PATH_NAME
    json_path = output_dir / SLICE_METRICS_JSON_NAME

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
        "min_class_support": MIN_SLICE_CLASS_SUPPORT,
        "rows": slice_df.to_dict(orient="records"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def export_mbert_slice_artifacts(
    test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    output_dir: Path,
    *,
    run_id: str,
    checkpoint: str,
    split: str,
    extra_metadata: dict[str, Any] | None = None,
    merge: bool = True,
) -> dict[str, Path]:
    """Calcula y exporta cortes mBERT + sonda dialectal para una corrida oficial."""
    slices = slice_metrics(test, y_true, y_pred, y_prob)
    slice_rows = slice_metrics_to_rows(
        slices,
        model="mbert",
        checkpoint=checkpoint,
        split=split,
        run_id=run_id,
        n_test=len(test),
    )
    slice_df = pd.DataFrame(slice_rows)
    perclass_rows = collect_perclass_rows(
        slices,
        model="mbert",
        checkpoint=checkpoint,
        split=split,
        run_id=run_id,
        n_test=len(test),
    )
    perclass_df = pd.DataFrame(perclass_rows)

    meta = {"run_id": run_id, "checkpoint": checkpoint, **(extra_metadata or {})}
    slice_csv, slice_json = export_slice_metrics(
        slice_df, output_dir, extra_metadata=meta, merge=merge
    )
    perclass_path = export_slice_perclass_metrics(
        perclass_df, output_dir, extra_metadata=meta, merge=merge
    )

    bias_df = dialect_bias_probe(
        test,
        y_true,
        y_pred,
        model="mbert",
        checkpoint=checkpoint,
        run_id=run_id,
        split=split,
    )
    bias_csv, bias_json = export_dialect_bias(
        bias_df, output_dir, extra_metadata=meta, merge=merge
    )

    return {
        "slice_metrics_csv": slice_csv,
        "slice_metrics_json": slice_json,
        "slice_perclass_csv": perclass_path,
        "dialect_bias_csv": bias_csv,
        "dialect_bias_json": bias_json,
    }
