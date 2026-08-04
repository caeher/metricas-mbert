#!/usr/bin/env python
"""Ejecuta benchmark zero-shot contra datasets públicos."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.config import load_settings
from discurso_odio.evaluation.benchmark import run_benchmark
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    settings = load_settings()
    model_dir = MODELS_DIR / "mbert-sv"
    if not model_dir.exists():
        logger.error("Modelo no encontrado en %s. Ejecute train_mbert.py primero.", model_dir)
        sys.exit(1)

    results = run_benchmark(str(model_dir))
    out_path = REPORTS_METRICS / "benchmark_externo.csv"
    results.to_csv(out_path, index=False)
    logger.info("Resultados guardados en %s", out_path)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
