#!/usr/bin/env python
"""Orquesta entrenamiento B2, evaluación test + batería 144 y decisión de adopción."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from discurso_odio.evaluation.adversarial import (
    build_battery_144_manifest,
    decide_mitigation_adoption_multiobjective,
    evaluate_adversarial_144_battery,
    evaluate_adversarial_battery,
    evaluate_test_metrics_parallel,
    export_before_after,
    load_battery_144_manifest,
)
from discurso_odio.evaluation.official_mbert import (
    evaluate_mbert_official,
    load_official_run_metadata,
    official_checkpoint_label,
)
from discurso_odio.models.mbert_artifacts import read_checkpoint_identity
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_train(output_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_mbert.py"),
        "--b2",
        "--exclude-adversarial-overlap",
        "--output-dir",
        str(output_dir),
    ]
    logger.info("Entrenando: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline B2 mBERT (informe verificación).")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv-b2",
    )
    parser.add_argument("--adopt", action="store_true", help="Promover candidato si pasa criterio")
    args = parser.parse_args()

    build_battery_144_manifest()
    load_battery_144_manifest()

    official = load_official_run_metadata()
    baseline_dir = args.baseline_dir
    candidate_dir = args.candidate_dir

    if baseline_dir.exists():
        baseline_75 = evaluate_adversarial_battery(
            model_dir=baseline_dir,
            model_name="mbert-v1",
            checkpoint=official_checkpoint_label(official, baseline_dir),
            run_id=official.get("run_id") if official else None,
            output_prefix="mbert_v1",
        )
        baseline_144 = evaluate_adversarial_144_battery(
            model_dir=baseline_dir,
            model_name="mbert-v1",
            checkpoint=baseline_75["checkpoint"],
            output_prefix="mbert_v1",
        )
        evaluate_test_metrics_parallel(
            model_dir=baseline_dir,
            model_name="mbert-v1",
            checkpoint=baseline_75["checkpoint"],
            output_prefix="mbert_v1",
        )

    if not args.skip_train:
        _run_train(candidate_dir)

    if not candidate_dir.exists():
        logger.error("Candidato no encontrado: %s", candidate_dir)
        sys.exit(1)

    identity = read_checkpoint_identity(candidate_dir)
    checkpoint = identity.get("checkpoint") or official_checkpoint_label(None, candidate_dir)

    cand_75 = evaluate_adversarial_battery(
        model_dir=candidate_dir,
        model_name="mbert-b2",
        checkpoint=str(checkpoint),
        output_prefix="mbert_b2",
    )
    cand_144 = evaluate_adversarial_144_battery(
        model_dir=candidate_dir,
        model_name="mbert-b2",
        checkpoint=str(checkpoint),
        output_prefix="mbert_b2",
    )
    test_metrics = evaluate_test_metrics_parallel(
        model_dir=candidate_dir,
        model_name="mbert-b2",
        checkpoint=str(checkpoint),
        output_prefix="mbert_b2",
    )

    before_rows = []
    if (REPORTS_METRICS / "mbert_v1_parallel_test_metrics.csv").exists():
        v1_test = pd.read_csv(REPORTS_METRICS / "mbert_v1_parallel_test_metrics.csv").iloc[0]
        v1_144_path = REPORTS_METRICS / "mbert_v1_144_adversarial_summary.json"
        v1_144 = json.loads(v1_144_path.read_text(encoding="utf-8")) if v1_144_path.exists() else {}
        before_rows.append(
            {
                "stage": "baseline_v1",
                "test_f1_macro": float(v1_test["f1_macro"]),
                "adv_global_correct": v1_144.get("global_correct"),
                "adv_threat_fn": v1_144.get("threat_fn"),
                "group_c_correct": None,
                "fpr_gap": v1_144.get("fpr_dialectal_gap"),
                "group_g_innocuous_fp": v1_144.get("group_g_innocuous_fp"),
                "battery_pred_class3": v1_144.get("battery_pred_class3"),
            }
        )

    before_rows.append(
        {
            "stage": "candidate_b2",
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "adv_global_correct": cand_144.get("global_correct"),
            "adv_threat_fn": cand_144.get("threat_fn"),
            "group_c_correct": None,
            "fpr_gap": cand_144.get("fpr_dialectal_gap"),
            "group_g_innocuous_fp": cand_144.get("group_g_innocuous_fp"),
            "battery_pred_class3": cand_144.get("battery_pred_class3"),
        }
    )
    comparison_df = pd.DataFrame(before_rows)
    comparison_path = REPORTS_METRICS / "mbert_b2_adoption_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    adopted = False
    notes = "Sin baseline v1 para comparar."
    if len(comparison_df) >= 2:
        adopted, notes = decide_mitigation_adoption_multiobjective(comparison_df)

    meta = {
        "mitigation_adopted": adopted,
        "adoption_notes": notes,
        "candidate_dir": str(candidate_dir),
        "test_metrics": test_metrics,
        "adversarial_75": cand_75,
        "adversarial_144": cand_144,
    }
    meta_path = REPORTS_METRICS / "mbert_b2_adoption_decision.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Adopción B2: %s — %s", adopted, notes)

    if adopted and args.adopt:
        logger.info("Promoviendo candidato a checkpoint oficial...")
        evaluate_mbert_official(model_dir=candidate_dir)
        logger.info("Evaluación oficial completada sobre %s", candidate_dir)


if __name__ == "__main__":
    main()
