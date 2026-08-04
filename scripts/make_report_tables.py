#!/usr/bin/env python
"""Consolida tablas de métricas en reports/tables/."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS, REPORTS_TABLES


def main():
    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)

    metrics_path = MODELS_DIR / "mbert-sv" / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        rows = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
        df = pd.DataFrame([rows])
        df.to_csv(REPORTS_TABLES / "mbert_test_metrics.md", index=False)
        with open(REPORTS_TABLES / "mbert_test_metrics.md", "w", encoding="utf-8") as f:
            f.write("# Métricas mBERT — Test\n\n")
            f.write(df.to_markdown(index=False))

    for csv in REPORTS_METRICS.glob("*.csv"):
        if csv.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        md_path = REPORTS_TABLES / (csv.stem + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {csv.stem}\n\n")
            f.write(df.to_markdown(index=False))
        print(f"Generado: {md_path}")


if __name__ == "__main__":
    main()
