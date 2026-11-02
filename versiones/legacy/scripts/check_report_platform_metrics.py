#!/usr/bin/env python
"""Verifica consistencia entre slice_metrics.csv y LaTeX generado por plataforma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.evaluation.report_consistency import assert_platform_consistency
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS

GENERATED_DIR = PROJECT_ROOT / "reports" / "latex" / "generated"


def main():
    parser = argparse.ArgumentParser(description="Verifica métricas por plataforma en el informe.")
    parser.add_argument(
        "--strict-xlmr",
        action="store_true",
        help="Falla si faltan predicciones XLM-R sin documentación en el informe",
    )
    args = parser.parse_args()

    assert_platform_consistency(
        REPORTS_METRICS / "slice_metrics.csv",
        GENERATED_DIR,
        strict_xlmr=args.strict_xlmr,
    )
    print("Consistencia de métricas por plataforma: OK")


if __name__ == "__main__":
    main()
