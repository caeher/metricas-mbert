#!/usr/bin/env python
"""Verifica consistencia de la métrica oficial mBERT en el informe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.evaluation.report_consistency import assert_mbert_official_consistency
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS

GENERATED_DIR = PROJECT_ROOT / "reports" / "latex" / "generated"


def main():
    parser = argparse.ArgumentParser(description="Verifica métrica oficial mBERT en LaTeX.")
    args = parser.parse_args()
    _ = args
    assert_mbert_official_consistency(REPORTS_METRICS / "mbert_test_metrics.csv", GENERATED_DIR)
    print("Consistencia de métrica oficial mBERT: OK")


if __name__ == "__main__":
    main()
