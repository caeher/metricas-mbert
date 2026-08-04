#!/usr/bin/env python
"""Genera figuras de evaluación y tablas finales."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR, REPORTS_FIGURES, REPORTS_METRICS, REPORTS_TABLES
from discurso_odio.utils.plotting import get_label_names, setup_plot_style

setup_plot_style()
LABELS = get_label_names()

test = load_split("test")
model_dir = MODELS_DIR / "mbert-sv"
y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir)
y_true = test["label"].values

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABELS, ax=ax, cmap="Blues")
plt.title("Matriz de confusión — mBERT (test)")
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / "04_confusion_matrix.png")
plt.close()

# Comparison table
rows = []
if (REPORTS_METRICS / "baselines_tfidf.csv").exists():
    bl = pd.read_csv(REPORTS_METRICS / "baselines_tfidf.csv")
    for _, r in bl.iterrows():
        rows.append({"model": r["model"], "f1_macro": r["f1_macro"]})

if (REPORTS_METRICS / "mbert_test_metrics.csv").exists():
    m = pd.read_csv(REPORTS_METRICS / "mbert_test_metrics.csv").iloc[0]
    rows.append({"model": "mBERT", "f1_macro": m["f1_macro"]})

if (REPORTS_METRICS / "trivial_baselines.csv").exists():
    tb = pd.read_csv(REPORTS_METRICS / "trivial_baselines.csv", index_col=0)
    for name, val in tb["F1-macro"].items():
        rows.append({"model": f"trivial_{name}", "f1_macro": val})

comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
comparison.to_csv(REPORTS_METRICS / "model_comparison.csv", index=False)

REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
with open(REPORTS_TABLES / "model_comparison.md", "w", encoding="utf-8") as f:
    f.write("# Comparación de modelos\n\n")
    f.write(comparison.to_markdown(index=False))

if (REPORTS_METRICS / "benchmark_externo.csv").exists():
    bench = pd.read_csv(REPORTS_METRICS / "benchmark_externo.csv")
    with open(REPORTS_TABLES / "benchmark_externo.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark externo\n\n")
        f.write(bench.to_markdown(index=False))
    ok = bench[bench["status"] == "ok"]
    if len(ok):
        fig, ax = plt.subplots(figsize=(12, 5))
        ok.plot(x="dataset", y=["f1_macro", "f1_binary"], kind="bar", ax=ax)
        ax.set_title("Benchmark zero-shot")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(REPORTS_FIGURES / "05_benchmark_externo.png")
        plt.close()

print("Figuras y tablas generadas.")
