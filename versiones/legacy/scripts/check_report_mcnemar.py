#!/usr/bin/env python
"""Verifica consistencia entre mcnemar_tests.csv y LaTeX generado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.evaluation.report_consistency import assert_mcnemar_consistency
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS

GENERATED_DIR = PROJECT_ROOT / "reports" / "latex" / "generated"


def main():
    parser = argparse.ArgumentParser(description="Verifica McNemar en el informe.")
    args = parser.parse_args()
    _ = args

    assert_mcnemar_consistency(REPORTS_METRICS / "mcnemar_tests.csv", GENERATED_DIR)
    print("Consistencia de McNemar: OK")


if __name__ == "__main__":
    main()
