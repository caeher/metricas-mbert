#!/usr/bin/env python
"""Genera figuras de evaluación, tablas finales y figuras del informe LaTeX."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.paths import MODELS_DIR, PROJECT_ROOT, REPORTS_FIGURES, REPORTS_METRICS, REPORTS_TABLES
from discurso_odio.utils.plotting import setup_plot_style

setup_plot_style()
REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
LATEX_FIGURES = PROJECT_ROOT / "reports" / "latex" / "figures"


def build_comparison_df() -> pd.DataFrame:
    rows = []
    bl_path = REPORTS_METRICS / "baselines_tfidf.csv"
    if bl_path.exists():
        bl = pd.read_csv(bl_path)
        if "model" not in bl.columns:
            bl = pd.read_csv(bl_path, index_col=0).reset_index()
            bl.columns = ["model", "f1_macro"]
        elif bl.columns[0].startswith("Unnamed"):
            bl = bl.rename(columns={bl.columns[0]: "model"})
        for _, r in bl.iterrows():
            rows.append({"model": r["model"], "f1_macro": r["f1_macro"]})

    m_path = REPORTS_METRICS / "mbert_test_metrics.csv"
    if m_path.exists():
        m = pd.read_csv(m_path).iloc[0]
        rows.append({"model": "mBERT", "f1_macro": m["f1_macro"]})

    tb_path = REPORTS_METRICS / "trivial_baselines.csv"
    if tb_path.exists():
        tb = pd.read_csv(tb_path, index_col=0)
        col = "F1-macro" if "F1-macro" in tb.columns else tb.columns[0]
        for name, val in tb[col].items():
            rows.append({"model": f"trivial_{name}", "f1_macro": val})

    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(REPORTS_METRICS / "model_comparison.csv", index=False)
    with open(REPORTS_TABLES / "model_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Comparación de modelos\n\n")
        f.write(comparison.to_markdown(index=False))
    return comparison


def plot_confusion_matrix():
    import os

    if os.environ.get("SKIP_CONFUSION_MATRIX", "").lower() in {"1", "true", "yes"}:
        existing = REPORTS_FIGURES / "04_confusion_matrix.png"
        if existing.exists():
            print("Omitiendo regeneración de 04_confusion_matrix.png (SKIP_CONFUSION_MATRIX)")
            return
    try:
        from sklearn.metrics import ConfusionMatrixDisplay

        from discurso_odio.data.loaders import load_split
        from discurso_odio.models.mbert import predict
        from discurso_odio.utils.plotting import get_label_names
    except ImportError as e:
        print(f"Omitiendo matriz de confusión: {e}")
        return

    model_dir = MODELS_DIR / "mbert-sv"
    if not model_dir.exists():
        print("Modelo mBERT no encontrado; se omite 04_confusion_matrix.png")
        return

    labels = get_label_names()
    test = load_split("test")
    y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir)
    y_true = test["label"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, ax=ax, cmap="Blues"
    )
    plt.title("Matriz de confusión — mBERT (test)")
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / "04_confusion_matrix.png")
    plt.close()
    print("Generado: 04_confusion_matrix.png")


def plot_benchmark():
    path = REPORTS_METRICS / "benchmark_externo.csv"
    if not path.exists():
        return
    bench = pd.read_csv(path)
    with open(REPORTS_TABLES / "benchmark_externo.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark externo\n\n")
        f.write(bench.to_markdown(index=False))
    ok = bench[bench["status"] == "ok"] if "status" in bench.columns else bench
    if len(ok) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_df = ok.copy()
    plot_df["dataset"] = plot_df["dataset"].astype(str).str.split("/").str[-1]
    plot_df.plot(x="dataset", y=["f1_macro", "f1_binary"], kind="bar", ax=ax)
    ax.set_title("Benchmark zero-shot")
    ax.set_ylabel("F1")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / "05_benchmark_externo.png")
    plt.close()
    print("Generado: 05_benchmark_externo.png")


def plot_model_comparison_bar(comparison: pd.DataFrame):
    if comparison.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2a6f97" if m == "mBERT" else "#8ecae6" for m in comparison["model"]]
    ax.barh(comparison["model"], comparison["f1_macro"], color=colors)
    ax.set_xlabel("F1-macro")
    ax.set_title("Comparación de modelos (test)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.9, comparison["f1_macro"].max() * 1.1))
    for i, v in enumerate(comparison["f1_macro"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / "06_model_comparison_bar.png")
    plt.close()
    print("Generado: 06_model_comparison_bar.png")


def plot_slice_f1():
    path = REPORTS_METRICS / "slice_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(0.4 + 0.5 * (df["f1_macro"] / df["f1_macro"].max()))
    ax.barh(df["slice"], df["f1_macro"], color=colors)
    ax.set_xlabel("F1-macro")
    ax.set_title("F1-macro por corte (slice)")
    ax.invert_yaxis()
    for i, v in enumerate(df["f1_macro"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / "07_slice_f1_heatmap.png")
    plt.close()
    print("Generado: 07_slice_f1_heatmap.png")


def plot_lime_weights():
    path = REPORTS_METRICS / "lime_global_weights.csv"
    out = REPORTS_FIGURES / "08_lime_global_weights.png"
    if not path.exists() or path.stat().st_size < 10:
        _placeholder_figure(out, "LIME global weights\nno disponibles")
        return
    df = pd.read_csv(path)
    if df.empty or "class" not in df.columns:
        _placeholder_figure(out, "LIME global weights\nvacíos")
        return

    classes = list(df["class"].unique())
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        top = df[df["class"] == cls].nlargest(8, "importance")
        ax.barh(top["token"].astype(str), top["importance"], color="#219ebc")
        ax.set_title(str(cls))
        ax.invert_yaxis()
    fig.suptitle("Importancias globales LIME (top tokens)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Generado: 08_lime_global_weights.png")


def plot_training_curve():
    out = REPORTS_FIGURES / "09_training_loss.png"
    state_paths = list((MODELS_DIR / "mbert-sv").glob("checkpoint-*/trainer_state.json"))
    root_state = MODELS_DIR / "mbert-sv" / "trainer_state.json"
    if root_state.exists():
        state_paths.append(root_state)
    if not state_paths:
        train_metrics = MODELS_DIR / "mbert-sv" / "train_metrics.json"
        if train_metrics.exists():
            with open(train_metrics, encoding="utf-8") as f:
                m = json.load(f)
            _placeholder_figure(
                out,
                f"Val F1-macro={m.get('eval_f1_macro', 'N/A'):.3f}\n"
                f"(sin historial de steps)",
            )
        else:
            _placeholder_figure(out, "Historial de entrenamiento\nno disponible")
        return

    state_path = max(state_paths, key=lambda p: p.stat().st_mtime)
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    history = state.get("log_history", [])
    train_epochs, train_loss = [], []
    eval_epochs, eval_f1, eval_loss = [], [], []
    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            train_epochs.append(entry.get("epoch", entry.get("step", 0)))
            train_loss.append(entry["loss"])
        if "eval_f1_macro" in entry:
            eval_epochs.append(entry.get("epoch", entry.get("step", 0)))
            eval_f1.append(entry["eval_f1_macro"])
            eval_loss.append(entry.get("eval_loss", np.nan))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    if train_loss:
        ax1.plot(train_epochs, train_loss, label="Train loss", color="#023047")
    if eval_loss:
        ax1.plot(eval_epochs, eval_loss, label="Eval loss", color="#fb8500", linestyle="--")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    if eval_f1:
        ax2.plot(eval_epochs, eval_f1, label="Eval F1-macro", color="#219ebc", marker="o")
    ax2.set_ylabel("F1-macro")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Curva de entrenamiento mBERT")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Generado: 09_training_loss.png")


def _placeholder_figure(path: Path, text: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="#555")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Placeholder: {path.name}")


def copy_to_latex_figures():
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    for png in REPORTS_FIGURES.glob("*.png"):
        shutil.copy2(png, LATEX_FIGURES / png.name)
    print(f"Figuras copiadas a {LATEX_FIGURES}")


def main():
    comparison = build_comparison_df()
    plot_confusion_matrix()
    plot_benchmark()
    plot_model_comparison_bar(comparison)
    plot_slice_f1()
    plot_lime_weights()
    plot_training_curve()
    copy_to_latex_figures()
    print("Figuras y tablas generadas.")


if __name__ == "__main__":
    main()
