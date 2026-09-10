"""Evaluación de gates multiobjetivo para adopción F1 0.80."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from discurso_odio.config import load_settings
from discurso_odio.evaluation.adversarial import (
    evaluate_adversarial_144_battery,
    evaluate_adversarial_battery,
)
from discurso_odio.evaluation.official_mbert import evaluate_mbert_official
from discurso_odio.models.mbert_artifacts import read_checkpoint_identity
from discurso_odio.paths import REPORTS_METRICS

GATES_JSON = REPORTS_METRICS / "f1_target_gates.json"

# Baselines de referencia (B2 oficial / v1)
BASELINE_B2_TEST_F1 = 0.7610
BASELINE_V1_TEST_F1 = 0.7777
BASELINE_V1_ADV75 = 45
BASELINE_B2_ADV75 = 64
BASELINE_V1_FPR_GAP = 0.86
BASELINE_B2_FPR_GAP = 0.07


def evaluate_gates(
    model_dir: str | Path | None = None,
    *,
    run_official: bool = True,
    use_existing_metrics: bool = False,
) -> dict[str, Any]:
    """Evalúa candidato contra gates primarios y secundarios."""
    settings = load_settings()
    target = float(settings["project"]["target_f1_macro"])
    model_dir = Path(model_dir) if model_dir else None

    official_run = None
    if model_dir and model_dir.exists() and run_official:
        official_run = evaluate_mbert_official(model_dir=model_dir)
    elif use_existing_metrics and (REPORTS_METRICS / "mbert_official_run.json").exists():
        official_run = json.loads(
            (REPORTS_METRICS / "mbert_official_run.json").read_text(encoding="utf-8")
        )

    metrics_path = REPORTS_METRICS / "mbert_test_metrics.csv"
    test_f1 = BASELINE_B2_TEST_F1
    if metrics_path.exists():
        test_f1 = float(pd.read_csv(metrics_path).iloc[0]["f1_macro"])

    checkpoint = "unknown"
    if model_dir and model_dir.exists():
        identity = read_checkpoint_identity(model_dir)
        checkpoint = identity.get("checkpoint", "unknown")

    adv75_correct = BASELINE_B2_ADV75
    adv144_correct = None
    fpr_gap = BASELINE_B2_FPR_GAP

    if model_dir and model_dir.exists():
        adv75 = evaluate_adversarial_battery(
            model_dir=model_dir,
            model_name="mbert-candidate",
            checkpoint=str(checkpoint),
            output_prefix="mbert_candidate",
        )
        adv144 = evaluate_adversarial_144_battery(
            model_dir=model_dir,
            model_name="mbert-candidate",
            checkpoint=str(checkpoint),
            output_prefix="mbert_candidate",
        )
        adv75_correct = int(adv75.get("global_correct", 0))
        adv144_correct = adv144.get("global_correct")
    else:
        before_after = REPORTS_METRICS / "mbert_adversarial_before_after.json"
        if before_after.exists():
            data = json.loads(before_after.read_text(encoding="utf-8"))
            for row in data.get("rows", []):
                if row.get("stage") == "official_b2":
                    adv75_correct = int(row.get("adv_global_correct", adv75_correct))
                    fpr_gap = float(row.get("fpr_gap", fpr_gap))
                    break
        adv144_path = REPORTS_METRICS / "mbert_b2_144_adversarial_summary.json"
        if adv144_path.exists():
            adv144_correct = json.loads(adv144_path.read_text(encoding="utf-8")).get("global_correct")

    dialect_path = REPORTS_METRICS / "dialect_bias_probe.csv"
    if dialect_path.exists():
        df = pd.read_csv(dialect_path)
        if "fpr_gap" in df.columns and len(df):
            fpr_gap = float(df.iloc[-1]["fpr_gap"])
    slice_path = REPORTS_METRICS / "slice_metrics.csv"
    facebook_f1 = None
    if slice_path.exists():
        slices = pd.read_csv(slice_path)
        fb = slices[
            (slices["model"] == "mbert")
            & (slices["slice"] == "plataforma_Facebook")
        ]
        if not fb.empty:
            facebook_f1 = float(fb.iloc[-1]["f1_macro"])

    adv75_correct = int(adv75_correct)
    primary_pass = test_f1 >= target
    secondary = {
        "adv75_not_below_v1": adv75_correct >= BASELINE_V1_ADV75,
        "adv75_at_least_b2": adv75_correct >= BASELINE_B2_ADV75,
        "fpr_gap_not_worse_than_b2": fpr_gap <= max(BASELINE_B2_FPR_GAP + 0.10, 0.15),
    }

    adopted = primary_pass and secondary["adv75_not_below_v1"]

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir) if model_dir else None,
        "checkpoint": str(checkpoint),
        "target_f1_macro": target,
        "test_f1_macro": test_f1,
        "primary_pass": primary_pass,
        "secondary_gates": secondary,
        "adopted": adopted,
        "adv75_correct": adv75_correct,
        "adv75_n": 75,
        "adv144_correct": adv144_correct,
        "fpr_dialectal_gap": fpr_gap,
        "facebook_f1_macro": facebook_f1,
        "official_run_id": official_run.get("run_id") if official_run else None,
        "notes": _gate_notes(primary_pass, test_f1, target, secondary, adv75_correct),
    }
    GATES_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATES_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _gate_notes(
    primary: bool,
    test_f1: float,
    target: float,
    secondary: dict[str, bool],
    adv75: int,
) -> str:
    parts = []
    if primary:
        parts.append(f"F1 test {test_f1:.4f} ≥ meta {target:.2f}.")
    else:
        parts.append(f"F1 test {test_f1:.4f} < meta {target:.2f} (gap {target - test_f1:.4f}).")
    parts.append(f"Adversarial 75: {adv75}/75.")
    if not secondary["adv75_not_below_v1"]:
        parts.append("Degradación adversarial por debajo de v1.")
    if secondary["fpr_gap_not_worse_than_b2"]:
        parts.append("Sesgo dialectal dentro de tolerancia B2.")
    return " ".join(parts)
