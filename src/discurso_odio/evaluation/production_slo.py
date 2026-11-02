"""Evaluación del contrato operativo production_slo_v1 (ISSUE-14)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from discurso_odio.evaluation.adversarial import (
    evaluate_adversarial_144_battery,
    evaluate_adversarial_battery,
)
from discurso_odio.models.mbert_artifacts import read_checkpoint_identity
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS

DEFAULT_SLO_PATH = REPORTS_METRICS / "production_slo_v1.json"
RESULT_JSON = REPORTS_METRICS / "production_slo_evaluation.json"
DEFAULT_ADVERSARIAL_OUTPUT_PREFIX = "mbert_slo"


def load_production_slo(path: Path | None = None) -> dict[str, Any]:
    slo_path = path or DEFAULT_SLO_PATH
    if not slo_path.is_file():
        raise FileNotFoundError(f"No existe contrato SLO: {slo_path}")
    with open(slo_path, encoding="utf-8") as handle:
        return json.load(handle)


def _latest_mbert_slice_rows(slice_df: pd.DataFrame) -> pd.DataFrame:
    mbert = slice_df[slice_df["model"] == "mbert"].copy()
    if mbert.empty:
        return mbert
    if "run_id" in mbert.columns:
        latest_run = mbert["run_id"].iloc[-1]
        mbert = mbert[mbert["run_id"] == latest_run]
    return mbert


def _slice_f1(slice_df: pd.DataFrame, slice_name: str) -> float | None:
    rows = slice_df[slice_df["slice"] == slice_name]
    if rows.empty:
        return None
    return float(rows.iloc[-1]["f1_macro"])


def _weighted_platform_f1(
    slice_df: pd.DataFrame,
    weights: dict[str, float],
) -> tuple[float | None, dict[str, float | None]]:
    values: dict[str, float | None] = {}
    for slice_name in weights:
        values[slice_name] = _slice_f1(slice_df, slice_name)
    present = {k: v for k, v in values.items() if v is not None}
    if not present:
        return None, values
    weight_sum = sum(weights[k] for k in present)
    if weight_sum <= 0:
        return None, values
    weighted = sum(present[k] * weights[k] for k in present) / weight_sum
    return float(weighted), values


def _min_required_slices(
    slice_df: pd.DataFrame,
    min_slice: dict[str, float],
) -> tuple[float | None, dict[str, float | None]]:
    actual: dict[str, float | None] = {}
    for slice_name in min_slice:
        actual[slice_name] = _slice_f1(slice_df, slice_name)
    present = [v for v in actual.values() if v is not None]
    if not present:
        return None, actual
    return float(min(present)), actual


def _class_recall_from_metrics(metrics_row: pd.Series, class_name: str) -> float | None:
    col = f"recall_{class_name}"
    if col not in metrics_row.index:
        return None
    value = metrics_row[col]
    if pd.isna(value):
        return None
    return float(value)


def _min_required_class_recall(
    metrics_row: pd.Series,
    min_recall: dict[str, float],
) -> tuple[float | None, dict[str, float | None]]:
    actual: dict[str, float | None] = {}
    for class_name in min_recall:
        actual[class_name] = _class_recall_from_metrics(metrics_row, class_name)
    present = [v for v in actual.values() if v is not None]
    if not present:
        return None, actual
    return float(min(present)), actual


def _load_adversarial_summary(output_prefix: str) -> dict[str, Any] | None:
    path = REPORTS_METRICS / f"{output_prefix}_adversarial_summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_adversarial_144_summary(output_prefix: str) -> dict[str, Any] | None:
    path = REPORTS_METRICS / f"{output_prefix}_144_adversarial_summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _fpr_dialectal_gap_from_probe(
    dialect_df: pd.DataFrame,
    *,
    checkpoint: str | None = None,
) -> float | None:
    """FPR(con_jerga) - FPR(sin_jerga) en No Tóxico sobre test oficial."""
    if dialect_df.empty or "group" not in dialect_df.columns:
        return None
    subset = dialect_df[dialect_df["model"] == "mbert"].copy()
    if checkpoint and "checkpoint" in subset.columns:
        matched = subset[subset["checkpoint"].astype(str) == str(checkpoint)]
        if not matched.empty:
            subset = matched
    if subset.empty:
        return None
    by_group = {
        str(row["group"]): float(row["fpr"])
        for _, row in subset.iterrows()
        if pd.notna(row.get("fpr"))
    }
    if "con_jerga" not in by_group or "sin_jerga" not in by_group:
        return None
    return by_group["con_jerga"] - by_group["sin_jerga"]


def _fpr_from_adversarial_summary(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    raw = summary.get("fpr_dialectal_gap")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return float(raw)


def evaluate_production_slo(
    *,
    model_dir: Path | None = None,
    slo_path: Path | None = None,
    run_adversarial: bool = True,
    use_existing_metrics: bool = True,
    output_path: Path | None = None,
    adversarial_output_prefix: str = DEFAULT_ADVERSARIAL_OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Calcula score operativo y cumplimiento del SLO v1."""
    slo = load_production_slo(slo_path)
    target = float(slo["target_operational_score"])
    weights = {str(k): float(v) for k, v in slo["platform_traffic_weights"].items()}
    min_slice = {str(k): float(v) for k, v in slo["min_slice_f1_macro"].items()}
    min_recall = {str(k): float(v) for k, v in slo["min_class_recall"].items()}
    constraints_cfg = slo["constraints"]
    adv75_min = int(constraints_cfg["adv75_min_correct"])
    adv75_n = int(constraints_cfg.get("adv75_n", 75))
    fpr_max = float(constraints_cfg["fpr_dialectal_gap_max"])

    metrics_path = REPORTS_METRICS / "mbert_test_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Falta {metrics_path}; ejecute evaluate_mbert_official primero.")
    metrics_row = pd.read_csv(metrics_path).iloc[0]
    test_f1_macro = float(metrics_row["f1_macro"])

    slice_path = REPORTS_METRICS / "slice_metrics.csv"
    if not slice_path.exists():
        raise FileNotFoundError(f"Falta {slice_path}; ejecute evaluación oficial o compute_slice_metrics.")
    slice_df = _latest_mbert_slice_rows(pd.read_csv(slice_path))

    weighted_f1, platform_actual = _weighted_platform_f1(slice_df, weights)
    min_slice_score, slice_actual = _min_required_slices(slice_df, min_slice)
    min_recall_score, recall_actual = _min_required_class_recall(metrics_row, min_recall)

    components: dict[str, float | None] = {
        "weighted_platform_f1_macro": weighted_f1,
        "min_required_slice_f1_macro": min_slice_score,
        "min_required_class_recall": min_recall_score,
    }
    score_values = [v for v in components.values() if v is not None]
    operational_score = float(min(score_values)) if score_values else None

    adv75_correct: int | None = None
    adv144_correct: int | None = None
    fpr_gap: float | None = None
    checkpoint = "unknown"
    adv_summary: dict[str, Any] | None = None
    if model_dir and model_dir.exists():
        identity = read_checkpoint_identity(model_dir)
        checkpoint = str(identity.get("checkpoint", "unknown"))
        if run_adversarial:
            adv75: dict[str, Any]
            try:
                adv75 = evaluate_adversarial_battery(
                    model_dir=model_dir,
                    model_name="mbert-slo",
                    checkpoint=checkpoint,
                    output_prefix=adversarial_output_prefix,
                )
            except ValueError as exc:
                if "Se esperaban" in str(exc) and "casos adversariales" in str(exc):
                    adv75 = {"global_correct": None}
                else:
                    raise
            try:
                adv144 = evaluate_adversarial_144_battery(
                    model_dir=model_dir,
                    model_name="mbert-slo",
                    checkpoint=checkpoint,
                    output_prefix=adversarial_output_prefix,
                )
            except ValueError as exc:
                if "Se esperaban" in str(exc) and "casos adversariales" in str(exc):
                    adv144 = {"global_correct": None}
                elif "SHA-256" in str(exc) and "test_adversarial_144" in str(exc):
                    adv144 = {"global_correct": None}
                else:
                    raise
            adv75_correct = (
                int(adv75.get("global_correct", 0))
                if adv75.get("global_correct") is not None
                else None
            )
            adv144_correct = adv144.get("global_correct")
            fpr_gap = _fpr_from_adversarial_summary(adv75)
            adv_summary = adv75
    if adv75_correct is None and use_existing_metrics:
        adv_summary = _load_adversarial_summary(adversarial_output_prefix)
        if adv_summary:
            adv75_correct = int(adv_summary.get("global_correct", 0))
            if fpr_gap is None:
                fpr_gap = _fpr_from_adversarial_summary(adv_summary)
        if adv144_correct is None:
            adv144_summary = _load_adversarial_144_summary(adversarial_output_prefix)
            if adv144_summary:
                adv144_correct = adv144_summary.get("global_correct")
        if adv75_correct is None:
            candidate_adv = REPORTS_METRICS / "mbert_candidate_adversarial_summary.json"
            if candidate_adv.exists():
                adv_summary = json.loads(candidate_adv.read_text(encoding="utf-8"))
                adv75_correct = int(adv_summary.get("global_correct", 0))
                if fpr_gap is None:
                    fpr_gap = _fpr_from_adversarial_summary(adv_summary)
        if adv75_correct is None and model_dir is None:
            before_after = REPORTS_METRICS / "mbert_adversarial_before_after.json"
            if before_after.exists():
                data = json.loads(before_after.read_text(encoding="utf-8"))
                for row in data.get("rows", []):
                    if row.get("stage") == "official_b2":
                        adv75_correct = int(row.get("adv_global_correct", 0))
                        break

    dialect_path = REPORTS_METRICS / "dialect_bias_probe.csv"
    if dialect_path.exists():
        dialect_df = pd.read_csv(dialect_path)
        if fpr_gap is None:
            fpr_gap = _fpr_dialectal_gap_from_probe(dialect_df, checkpoint=checkpoint)
        if fpr_gap is None and "fpr_gap" in dialect_df.columns and len(dialect_df):
            fpr_gap = float(dialect_df.iloc[-1]["fpr_gap"])
        if fpr_gap is None and adv_summary is None:
            adv_summary = _load_adversarial_summary(adversarial_output_prefix)
        if fpr_gap is None:
            fpr_gap = _fpr_from_adversarial_summary(adv_summary)

    constraints_evaluated = adv75_correct is not None and fpr_gap is not None

    constraints_pass = {
        "adv75_min_correct": adv75_correct is not None and adv75_correct >= adv75_min,
        "fpr_dialectal_gap_max": fpr_gap is not None and fpr_gap <= fpr_max,
    }
    if adv75_correct is None:
        constraints_pass["adv75_min_correct"] = False
    if fpr_gap is None:
        constraints_pass["fpr_dialectal_gap_max"] = False

    slice_threshold_pass = {
        name: (slice_actual.get(name) is not None and slice_actual[name] >= threshold)
        for name, threshold in min_slice.items()
    }
    recall_threshold_pass = {
        name: (recall_actual.get(name) is not None and recall_actual[name] >= threshold)
        for name, threshold in min_recall.items()
    }

    slo_pass = (
        operational_score is not None
        and operational_score >= target
        and all(constraints_pass.values())
        and all(slice_threshold_pass.values())
        and all(recall_threshold_pass.values())
    )

    gaps: dict[str, Any] = {}
    if operational_score is not None:
        gaps["operational_score_to_target"] = round(target - operational_score, 4)
    for name, threshold in min_slice.items():
        actual = slice_actual.get(name)
        if actual is not None:
            gaps[f"slice_{name}"] = round(threshold - actual, 4)
    for name, threshold in min_recall.items():
        actual = recall_actual.get(name)
        if actual is not None:
            gaps[f"recall_{name}"] = round(threshold - actual, 4)

    slo_file = slo_path or DEFAULT_SLO_PATH
    try:
        slo_path_str = str(slo_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        slo_path_str = str(slo_file)

    payload: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "slo_id": slo["slo_id"],
        "slo_path": slo_path_str,
        "target_operational_score": target,
        "operational_score": round(operational_score, 4) if operational_score is not None else None,
        "slo_pass": slo_pass,
        "test_f1_macro": round(test_f1_macro, 4),
        "components": {k: (round(v, 4) if v is not None else None) for k, v in components.items()},
        "platform_f1_macro": {
            k: (round(v, 4) if v is not None else None) for k, v in platform_actual.items()
        },
        "slice_f1_macro": {
            k: (round(v, 4) if v is not None else None) for k, v in slice_actual.items()
        },
        "class_recall": {
            k: (round(v, 4) if v is not None else None) for k, v in recall_actual.items()
        },
        "slice_threshold_pass": slice_threshold_pass,
        "recall_threshold_pass": recall_threshold_pass,
        "constraints": {
            "adv75_correct": adv75_correct,
            "adv75_min_correct": adv75_min,
            "adv75_n": adv75_n,
            "adv144_correct": adv144_correct,
            "fpr_dialectal_gap": fpr_gap,
            "fpr_dialectal_gap_max": fpr_max,
            "pass": constraints_pass,
        },
        "gaps_to_slo": gaps,
        "adversarial_output_prefix": adversarial_output_prefix,
        "constraints_evaluated": constraints_evaluated,
        "model_dir": str(model_dir) if model_dir else None,
        "checkpoint": checkpoint,
        "notes": _slo_notes(
            operational_score,
            target,
            slo_pass,
            components,
            constraints_pass,
        ),
    }
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_path = output_path or RESULT_JSON
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _slo_notes(
    operational_score: float | None,
    target: float,
    slo_pass: bool,
    components: dict[str, float | None],
    constraints_pass: dict[str, bool],
) -> str:
    if operational_score is None:
        return "No se pudo calcular score operativo (faltan métricas de slice o recall)."
    parts = [
        f"Score operativo {operational_score:.4f} (meta {target:.2f}).",
        f"Cuello de botella: {_bottleneck(components)}.",
    ]
    if slo_pass:
        parts.append("Cumple production_slo_v1.")
    else:
        if operational_score < target:
            parts.append(f"Gap score {target - operational_score:.4f}.")
        if not all(constraints_pass.values()):
            parts.append("Restricciones adversarial/FPR no cumplidas.")
    return " ".join(parts)


def _bottleneck(components: dict[str, float | None]) -> str:
    present = {k: v for k, v in components.items() if v is not None}
    if not present:
        return "desconocido"
    key = min(present, key=present.get)
    return f"{key}={present[key]:.4f}"
