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

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.evaluation.slices import platform_rows
from discurso_odio.models.mbert import default_mbert_model_dir
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

    xlmr_pred_path = REPORTS_METRICS / "xlmr_test_predictions.csv"
    if xlmr_pred_path.exists():
        from sklearn.metrics import f1_score

        from discurso_odio.data.loaders import load_split
        from discurso_odio.evaluation.predictions import XLMR_PRED_COL, load_aligned_predictions

        test = load_split("test")
        y_pred = load_aligned_predictions(test, xlmr_pred_path, XLMR_PRED_COL, "XLM-R")
        f1 = f1_score(test["label"].values, y_pred, average="macro")
        rows.append({"model": "XLM-RoBERTa", "f1_macro": f1})

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

    model_dir = default_mbert_model_dir()
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


def _slice_label(slice_name: str) -> str:
    labels = {
        "plataforma_X": "Plataforma X",
        "plataforma_Facebook": "Plataforma Facebook",
        "con_jerga": "Con jerga",
        "sin_jerga": "Sin jerga",
        "sarcasmo_si": "Sarcasmo sí",
        "sarcasmo_no": "Sarcasmo no",
        "ironia_si": "Ironía sí",
        "ironia_no": "Ironía no",
        "contexto_politico_si": "Contexto político sí",
        "contexto_politico_no": "Contexto político no",
    }
    return labels.get(slice_name, slice_name.replace("_", " ").title())


def _slice_sort_key(slice_name: str) -> tuple[int, str]:
    order = [
        "plataforma_X",
        "plataforma_Facebook",
        "con_jerga",
        "sin_jerga",
        "sarcasmo_si",
        "sarcasmo_no",
        "ironia_si",
        "ironia_no",
        "contexto_politico_si",
        "contexto_politico_no",
    ]
    if slice_name in order:
        return (order.index(slice_name), slice_name)
    return (len(order), slice_name)


def plot_slice_f1(model: str | None = "mbert", outfile: str = "07_slice_f1_heatmap.png"):
    path = REPORTS_METRICS / "slice_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    if "model" in df.columns:
        if model and model in df["model"].values:
            df = df[df["model"] == model]
        elif "model" in df.columns:
            df = df[df["model"] == df["model"].iloc[0]]

    df = df.copy()
    df["label"] = df["slice"].map(_slice_label)
    if "n" in df.columns:
        df["label"] = df.apply(
            lambda r: f"{r['label']} (n={int(r['n'])})" if pd.notna(r.get("n")) else r["label"],
            axis=1,
        )
    df = df.sort_values("slice", key=lambda s: s.map(lambda x: _slice_sort_key(x)[0]))

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.Blues(0.35 + 0.5 * (df["f1_macro"] / max(df["f1_macro"].max(), 1e-6)))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["f1_macro"], color=colors, xerr=None)

    if {"f1_ci_lo", "f1_ci_hi"}.issubset(df.columns):
        has_ci = df["f1_ci_lo"].notna() & df["f1_ci_hi"].notna()
        if has_ci.any():
            xerr_lo = df.loc[has_ci, "f1_macro"] - df.loc[has_ci, "f1_ci_lo"]
            xerr_hi = df.loc[has_ci, "f1_ci_hi"] - df.loc[has_ci, "f1_macro"]
            ax.errorbar(
                df.loc[has_ci, "f1_macro"],
                np.where(has_ci)[0],
                xerr=[xerr_lo, xerr_hi],
                fmt="none",
                ecolor="#333333",
                capsize=3,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("F1-macro")
    model_labels = {"mbert": "mBERT", "xlm-roberta": "XLM-RoBERTa"}
    title_model = model_labels.get(model or "", model or "")
    ax.set_title(f"F1-macro por corte (slice){' — ' + title_model if title_model else ''}")
    ax.invert_yaxis()
    for i, v in enumerate(df["f1_macro"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / outfile)
    plt.close()
    print(f"Generado: {outfile}")


def plot_slice_f1_xlmr():
    path = REPORTS_METRICS / "slice_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "model" not in df.columns or "xlm-roberta" not in df["model"].values:
        return
    plot_slice_f1(model="xlm-roberta", outfile="07b_slice_f1_xlmr.png")


def plot_platform_f1_grouped():
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = REPORTS_FIGURES / "08_platform_f1_grouped.png"
    if not path.exists():
        _placeholder_figure(out, "slice_metrics.csv\nno disponible")
        return

    df = platform_rows(pd.read_csv(path))
    if df.empty:
        _placeholder_figure(out, "Sin métricas por plataforma")
        return

    platform_order = ["plataforma_X", "plataforma_Facebook"]
    platform_labels = {"plataforma_X": "X", "plataforma_Facebook": "Facebook"}
    model_labels = {"mbert": "mBERT", "xlm-roberta": "XLM-RoBERTa"}
    models = sorted(df["model"].unique()) if "model" in df.columns else ["mbert"]

    x = np.arange(len(platform_order))
    width = 0.35 if len(models) > 1 else 0.5
    colors = ["#219ebc", "#fb8500", "#8ecae6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("slice")
        f1s = [sub.loc[p, "f1_macro"] if p in sub.index else np.nan for p in platform_order]
        ns = [int(sub.loc[p, "n"]) if p in sub.index else 0 for p in platform_order]
        offsets = x + (i - (len(models) - 1) / 2) * width
        label = model_labels.get(model, model)
        bars = ax.bar(offsets, f1s, width=width, label=label, color=colors[i % len(colors)])
        for bar, n_val, f1_val in zip(bars, ns, f1s):
            if not np.isnan(f1_val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{f1_val:.3f}\n(n={n_val})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([platform_labels[p] for p in platform_order])
    ax.set_ylabel("F1-macro")
    ax.set_xlabel("Plataforma")
    ax.set_title("F1-macro por plataforma y modelo")
    ax.set_ylim(0, 1.05)
    if len(models) > 1:
        ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Generado: 08_platform_f1_grouped.png")


def plot_dialect_bias_fpr():
    path = REPORTS_METRICS / "dialect_bias_probe.csv"
    out = REPORTS_FIGURES / "10_dialect_bias_fpr.png"
    if not path.exists():
        _placeholder_figure(out, "dialect_bias_probe.csv\nno disponible")
        return

    df = pd.read_csv(path)
    if df.empty or "group" not in df.columns:
        _placeholder_figure(out, "dialect_bias_probe.csv\nformato inválido")
        return

    group_labels = {"con_jerga": "Con jerga", "sin_jerga": "Sin jerga"}
    models = list(df["model"].unique())
    groups = ["sin_jerga", "con_jerga"]
    x = np.arange(len(groups))
    width = 0.35 if len(models) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#219ebc", "#fb8500", "#8ecae6"]

    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("group")
        fprs = [sub.loc[g, "fpr"] if g in sub.index else np.nan for g in groups]
        ci_lo = [sub.loc[g, "fpr_ci_lo"] if g in sub.index else np.nan for g in groups]
        ci_hi = [sub.loc[g, "fpr_ci_hi"] if g in sub.index else np.nan for g in groups]
        offsets = x + (i - (len(models) - 1) / 2) * width
        label = "mBERT" if model == "mbert" else model
        bars = ax.bar(offsets, fprs, width=width, label=label, color=colors[i % len(colors)])
        yerr_lo = [f - lo if pd.notna(f) and pd.notna(lo) else 0 for f, lo in zip(fprs, ci_lo)]
        yerr_hi = [hi - f if pd.notna(f) and pd.notna(hi) else 0 for f, hi in zip(fprs, ci_hi)]
        ax.errorbar(
            offsets,
            fprs,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="#333333",
            capsize=4,
        )
        for bar, fpr in zip(bars, fprs):
            if pd.notna(fpr):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{fpr:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([group_labels[g] for g in groups])
    ax.set_ylabel("FPR (No Tóxico)")
    ax.set_ylim(0, min(1.0, max(df["fpr_ci_hi"].max(skipna=True) * 1.2, 0.5)))
    ax.set_title("Sonda de sesgo dialectal — FPR por grupo de jerga")
    if len(models) > 1:
        ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Generado: 10_dialect_bias_fpr.png")


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
    model_dir = default_mbert_model_dir()
    state_paths = list(model_dir.glob("checkpoint-*/trainer_state.json"))
    root_state = model_dir / "trainer_state.json"
    if root_state.exists():
        state_paths.append(root_state)
    if not state_paths:
        train_metrics = model_dir / "train_metrics.json"
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


def plot_adversarial_families():
    path = REPORTS_METRICS / "mbert_adversarial_family_metrics.csv"
    out = REPORTS_FIGURES / "11_adversarial_families.png"
    if not path.exists():
        _placeholder_figure(out, "mbert_adversarial_family_metrics.csv\nno disponible")
        return

    df = pd.read_csv(path)
    sub = df[df["model"] == "mbert"] if "mbert" in df["model"].values else df
    keys = ["A", "B", "C", "D", "E"]
    labels = {
        "A": "Ofuscación",
        "B": "Emojis",
        "C": "Amenazas",
        "D": "Jerga SV",
        "E": "Control",
    }
    mbert_scores = []
    xlmr_scores = []
    baseline = {
        "A": 17 / 20,
        "B": 6 / 18,
        "C": 5 / 11,
        "D": 1 - 6 / 14,
        "E": 1 - 0 / 12,
    }
    for key in keys:
        row = sub[sub["family_key"] == key]
        if row.empty:
            mbert_scores.append(np.nan)
        elif row.iloc[0]["metric_type"] == "fpr":
            mbert_scores.append(1 - float(row.iloc[0]["fpr"]))
        else:
            mbert_scores.append(float(row.iloc[0]["correct"]) / float(row.iloc[0]["n"]))
        xlmr_scores.append(baseline.get(key, np.nan))

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, mbert_scores, width, label="mBERT (oficial)", color="#219ebc")
    ax.bar(x + width / 2, xlmr_scores, width, label="XLM-R v1 (ref.)", color="#fb8500", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys])
    ax.set_ylabel("Tasa de acierto / (1 - FPR)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Batería adversarial por familia (diagnóstico)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Generado: 11_adversarial_families.png")


def main():
    comparison = build_comparison_df()
    plot_confusion_matrix()
    plot_benchmark()
    plot_model_comparison_bar(comparison)
    plot_slice_f1()
    plot_slice_f1_xlmr()
    plot_platform_f1_grouped()
    plot_dialect_bias_fpr()
    plot_adversarial_families()
    plot_lime_weights()
    plot_training_curve()
    copy_to_latex_figures()
    print("Figuras y tablas generadas.")


if __name__ == "__main__":
    main()
