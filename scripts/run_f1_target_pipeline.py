#!/usr/bin/env python
"""Pipeline integral F1-macro 0.80 (Fases 0–4 + gates oficiales)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.config import load_settings
from discurso_odio.evaluation.ensemble import export_ensemble_manifest
from discurso_odio.evaluation.error_analysis import export_error_analysis
from discurso_odio.evaluation.f1_gates import evaluate_gates
from discurso_odio.evaluation.thresholds import tune_thresholds_on_val
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS = Path(__file__).resolve().parent


def _run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    logger.info("Ejecutando: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline F1-macro 0.80 mBERT")
    parser.add_argument("--skip-error-analysis", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-dapt", action="store_true")
    parser.add_argument("--skip-thresholds", action="store_true")
    parser.add_argument("--run-dapt", action="store_true", help="Forzar Fase 4 DAPT")
    parser.add_argument("--sweep-max-runs", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv-f1",
    )
    parser.add_argument(
        "--ensemble-dirs",
        nargs="*",
        default=None,
        help="Checkpoints para ensamble (default: output-dir + mbert-sv-b2 + mbert-sv)",
    )
    parser.add_argument("--eval-test-ensemble", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    target = settings["project"]["target_f1_macro"]
    logger.info("Meta F1-macro: %.2f", target)

    # Fase 0
    if not args.skip_error_analysis:
        logger.info("=== Fase 0: análisis de errores ===")
        report = export_error_analysis()
        logger.info(
            "Errores: %d/%d | F1=%.4f | adyacentes=%.1f%%",
            report["n_errors"],
            report["n_test"],
            report["f1_macro"],
            report["adjacent_error_pct"],
        )

    dapt_dir = MODELS_DIR / "mbert-sv-dapt"
    init_dir: Path | None = None

    # Fase 4 DAPT (opcional / forzado)
    if args.run_dapt and not args.skip_dapt:
        logger.info("=== Fase 4: DAPT MLM ===")
        _run("train_mbert_dapt.py", "--output-dir", str(dapt_dir))
        init_dir = dapt_dir

    # Fase 1: receta recomendada
    if not args.skip_train:
        logger.info("=== Fase 1: entrenamiento receta F1 ===")
        train_args = [
            "--b2",
            "--exclude-adversarial-overlap",
            "--output-dir",
            str(args.output_dir),
            "--synthetic-policy",
            "b2_no_template_heavy",
            "--synthetic-effective-fraction",
            "0.35",
            "--platform-class-boost",
            "default",
            "--label-smoothing",
            "0.0",
            "--loss-type",
            "ordinal",
            "--layerwise-lr-decay",
            "0.95",
            "--epochs",
            "6",
            "--early-stopping-patience",
            "3",
        ]
        if init_dir:
            train_args.extend(["--init-model-dir", str(init_dir)])
        _run("train_mbert.py", *train_args)

    # Fase 2: barrido reducido
    if not args.skip_sweep:
        logger.info("=== Fase 2: barrido HP (val) ===")
        sweep_args = [
            "--max-runs",
            str(args.sweep_max_runs),
            "--seeds",
            "42,7",
            "--synthetic-policies",
            "b2_no_template_heavy,b2_full",
            "--loss-types",
            "ce,ordinal",
        ]
        if init_dir:
            sweep_args.extend(["--init-model-dir", str(init_dir)])
        _run("sweep_mbert_f1.py", *sweep_args)

        best_path = REPORTS_METRICS / "mbert_sweep_best.json"
        if best_path.exists():
            best = json.loads(best_path.read_text(encoding="utf-8"))
            logger.info("Mejor barrido val F1: %.4f (%s)", best["best_val_f1_macro"], best["best_run_id"])

    # Fase 3: umbrales
    if not args.skip_thresholds:
        logger.info("=== Fase 3: umbrales en val ===")
        model_dir = args.output_dir
        if model_dir.exists():
            tune_thresholds_on_val(model_dir)
        else:
            logger.warning("Sin checkpoint en %s; omitiendo umbrales", model_dir)

    # Fase 3: ensamble
    ensemble_dirs = args.ensemble_dirs
    if ensemble_dirs is None:
        candidates = [args.output_dir, MODELS_DIR / "mbert-sv-b2", MODELS_DIR / "mbert-sv"]
        ensemble_dirs = [str(p) for p in candidates if Path(p).exists()]
    if len(ensemble_dirs) >= 2:
        logger.info("=== Fase 3: ensamble ===")
        export_ensemble_manifest(ensemble_dirs)
        if args.eval_test_ensemble:
            _run(
                "ensemble_mbert.py",
                "--model-dirs",
                *ensemble_dirs,
                "--eval-test",
            )

    # Gates oficiales
    logger.info("=== Gates oficiales ===")
    eval_dir = args.output_dir if args.output_dir.exists() else MODELS_DIR / "mbert-sv-b2"
    use_existing = not eval_dir.exists()
    if use_existing:
        logger.warning("Sin checkpoint local; evaluando gates con métricas oficiales existentes")

    gates = evaluate_gates(
        eval_dir if eval_dir.exists() else None,
        use_existing_metrics=use_existing,
        run_official=eval_dir.exists(),
    )
    logger.info("Resultado gates: %s", gates["notes"])
    logger.info("Adoptado: %s | test F1=%.4f", gates["adopted"], gates["test_f1_macro"])

    summary_path = REPORTS_METRICS / "f1_target_pipeline_summary.json"
    summary_path.write_text(json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resumen: %s", summary_path)

    if not gates["primary_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
