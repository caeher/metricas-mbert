#!/usr/bin/env python
"""Genera tablas LaTeX en reports/latex/generated/ a partir de reports/metrics/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.paths import MODELS_DIR, PROJECT_ROOT, REPORTS_FIGURES, REPORTS_METRICS

LATEX_DIR = PROJECT_ROOT / "reports" / "latex"
GENERATED_DIR = LATEX_DIR / "generated"
LATEX_FIGURES = LATEX_DIR / "figures"


def latex_escape(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return latex_escape(value)


def write_tabular(path: Path, headers: list[str], rows: list[list[str]], col_spec: str | None = None):
    if col_spec is None:
        col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado: {path.name}")


def placeholder(path: Path, message: str = "Datos no disponibles"):
    write_tabular(path, ["Nota"], [[latex_escape(message)]])


def gen_trivial_baselines():
    path = REPORTS_METRICS / "trivial_baselines.csv"
    out = GENERATED_DIR / "tab_trivial_baselines.tex"
    if not path.exists() or path.stat().st_size < 5:
        placeholder(out)
        return
    df = pd.read_csv(path, index_col=0)
    rows = [[latex_escape(idx), fmt_num(val)] for idx, val in df.iloc[:, 0].items()]
    write_tabular(out, ["Baseline", "F1-macro"], rows)


def gen_class_weights():
    path = REPORTS_METRICS / "class_weights.csv"
    out = GENERATED_DIR / "tab_class_weights.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    rows = [
        [latex_escape(r["clase"]), fmt_num(r["peso_calculado"]), fmt_num(r["peso_informe"])]
        for _, r in df.iterrows()
    ]
    write_tabular(out, ["Clase", "Peso calculado", "Peso informe"], rows)


def gen_baselines():
    path = REPORTS_METRICS / "baselines_tfidf.csv"
    out = GENERATED_DIR / "tab_baselines.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    if "model" not in df.columns:
        df = pd.read_csv(path, index_col=0).reset_index()
        df.columns = ["model", "f1_macro"]
    elif str(df.columns[0]).startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "model"})
    rows = [[latex_escape(r["model"]), fmt_num(r["f1_macro"])] for _, r in df.iterrows()]
    write_tabular(out, ["Modelo", "F1-macro"], rows)


def gen_model_comparison():
    path = REPORTS_METRICS / "model_comparison.csv"
    out = GENERATED_DIR / "tab_model_comparison.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path).sort_values("f1_macro", ascending=False)
    rows = [[latex_escape(r["model"]), fmt_num(r["f1_macro"])] for _, r in df.iterrows()]
    write_tabular(out, ["Modelo", "F1-macro"], rows)


def gen_mbert_tables():
    path = REPORTS_METRICS / "mbert_test_metrics.csv"
    out_g = GENERATED_DIR / "tab_mbert_global.tex"
    out_p = GENERATED_DIR / "tab_mbert_perclass.tex"
    if not path.exists():
        placeholder(out_g)
        placeholder(out_p)
        return
    row = pd.read_csv(path).iloc[0]
    global_keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1_macro", "F1-macro"),
        ("f1_weighted", "F1-weighted"),
        ("mcc", "MCC"),
        ("qwk", "QWK"),
        ("kappa_linear", "Kappa lineal"),
        ("ordinal_mae", "Ordinal MAE"),
        ("ordinal_rmse", "Ordinal RMSE"),
        ("large_error_pct", "Errores grandes (\\%)"),
        ("roc_auc_ovr_macro", "ROC-AUC OVR"),
        ("log_loss", "Log-loss"),
        ("ece", "ECE"),
        ("brier_macro", "Brier macro"),
        ("f1_bootstrap_lo", "F1 bootstrap lo"),
        ("f1_bootstrap_hi", "F1 bootstrap hi"),
    ]
    rows_g = [[label, fmt_num(row[key])] for key, label in global_keys if key in row.index]
    write_tabular(out_g, ["Métrica", "Valor"], rows_g)

    classes = ["No Tóxico", "Lenguaje Ofensivo", "Discurso de Odio", "Amenazas/Violencia"]
    rows_p = []
    for cls in classes:
        rows_p.append(
            [
                latex_escape(cls),
                fmt_num(row.get(f"precision_{cls}")),
                fmt_num(row.get(f"recall_{cls}")),
                fmt_num(row.get(f"f1_{cls}")),
                fmt_num(row.get(f"support_{cls}"), digits=0),
            ]
        )
    write_tabular(out_p, ["Clase", "Precision", "Recall", "F1", "Support"], rows_p)


def gen_slices():
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = GENERATED_DIR / "tab_slices.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    rows = [
        [latex_escape(r["slice"]), fmt_num(r["f1_macro"]), fmt_num(r["accuracy"])]
        for _, r in df.iterrows()
    ]
    write_tabular(out, ["Corte", "F1-macro", "Accuracy"], rows)


def gen_mcnemar():
    path = REPORTS_METRICS / "mcnemar_tests.csv"
    out = GENERATED_DIR / "tab_mcnemar.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    rows = [
        [
            latex_escape(r["baseline"]),
            fmt_num(r["statistic"]),
            fmt_num(r["p_value"], digits=6),
            fmt_num(r["b"], digits=0),
            fmt_num(r["c"], digits=0),
        ]
        for _, r in df.iterrows()
    ]
    write_tabular(out, ["Baseline", "Estadístico", "p-valor", "b", "c"], rows)


def _pct_latex(value: float, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    pct = float(value) * 100
    if digits == 0:
        return f"{pct:.0f}\\,\\%"
    return f"{pct:.{digits}f}\\,\\%"


def _frac_latex(fp: float, n: float) -> str:
    if pd.isna(fp) or pd.isna(n):
        return "---"
    return f"{int(fp)}/{int(n)}"


def gen_dialect_bias():
    path = REPORTS_METRICS / "dialect_bias_probe.csv"
    out = GENERATED_DIR / "tab_dialect_bias.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    if df.empty or "group" not in df.columns:
        placeholder(out, "Formato de dialect_bias_probe.csv no reconocido")
        return

    group_labels = {"con_jerga": "Con jerga", "sin_jerga": "Sin jerga"}
    rows = []
    for _, r in df.iterrows():
        model = str(r.get("model", "unknown"))
        group = group_labels.get(str(r["group"]), str(r["group"]))
        rows.append(
            [
                latex_escape(model),
                latex_escape(group),
                _frac_latex(r["fp"], r["n"]),
                fmt_num(r["fpr"]),
                fmt_num(r["fpr_ci_lo"]),
                fmt_num(r["fpr_ci_hi"]),
                fmt_num(r["n"], digits=0),
            ]
        )
    write_tabular(
        out,
        ["Modelo", "Grupo", "FP/n", "FPR", "IC 95\\% lo", "IC 95\\% hi", "n"],
        rows,
        col_spec="llccccr",
    )


def gen_dialect_bias_prose():
    """Genera párrafo LaTeX con cifras derivadas del artefacto (sin transcribir a mano)."""
    path = REPORTS_METRICS / "dialect_bias_probe.csv"
    out = GENERATED_DIR / "dialect_bias_prose.tex"
    if not path.exists():
        out.write_text(
            "% dialect_bias_probe.csv no disponible\n",
            encoding="utf-8",
        )
        return

    df = pd.read_csv(path)
    if df.empty:
        out.write_text("% dialect_bias_probe.csv vacío\n", encoding="utf-8")
        return

    # Preferir mBERT para el informe principal; si solo hay XLM-R, usar ese.
    model_order = ["mbert", "xlm-roberta"]
    model = next((m for m in model_order if m in df["model"].values), df["model"].iloc[0])
    sub = df[df["model"] == model]

    def _row(group: str) -> pd.Series | None:
        matches = sub[sub["group"] == group]
        return matches.iloc[0] if len(matches) else None

    con = _row("con_jerga")
    sin = _row("sin_jerga")
    if con is None or sin is None:
        out.write_text("% grupos de jerga incompletos en dialect_bias_probe.csv\n", encoding="utf-8")
        return

    model_label = "mBERT" if model == "mbert" else "XLM-RoBERTa"
    lines = [
        f"La Tabla~\\ref{{tab:dialect}} reporta la tasa de falsos positivos sobre No Tóxico "
        f"para {model_label}, calculada desde \\texttt{{reports/metrics/dialect\\_bias\\_probe.csv}}.",
        f"Con jerga local el FPR es \\textbf{{{fmt_num(con['fpr'])}}} "
        f"({_frac_latex(con['fp'], con['n'])}; $n={int(con['n'])}$), frente a "
        f"\\textbf{{{fmt_num(sin['fpr'])}}} "
        f"({_frac_latex(sin['fp'], sin['n'])}; $n={int(sin['n'])}$) sin jerga.",
        "Aunque el subconjunto con jerga es pequeño, el patrón sugiere que el modelo "
        "puede asociar léxico dialectal con toxicidad, un riesgo relevante para la equidad "
        "en el contexto salvadoreño.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_dialect_bias_conclusions():
    """Snippet breve para el capítulo de conclusiones."""
    path = REPORTS_METRICS / "dialect_bias_probe.csv"
    out = GENERATED_DIR / "dialect_bias_conclusions.tex"
    if not path.exists():
        out.write_text(
            "% dialect_bias_probe.csv no disponible\n",
            encoding="utf-8",
        )
        return

    df = pd.read_csv(path)
    model = "mbert" if "mbert" in df["model"].values else df["model"].iloc[0]
    sub = df[df["model"] == model]
    con = sub[sub["group"] == "con_jerga"].iloc[0]
    sin = sub[sub["group"] == "sin_jerga"].iloc[0]

    line = (
        f"  \\item La sonda dialectal muestra FPR elevado en No Tóxico con jerga local "
        f"({_pct_latex(con['fpr'])} vs.\\ {_pct_latex(sin['fpr'])}), "
        f"señal de posible sesgo léxico (fuente: "
        f"\\texttt{{reports/metrics/dialect\\_bias\\_probe.csv}})."
    )
    out.write_text(line + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_benchmark():
    path = REPORTS_METRICS / "benchmark_externo.csv"
    out = GENERATED_DIR / "tab_benchmark.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    rows = []
    for _, r in df.iterrows():
        name = str(r["dataset"]).split("/")[-1]
        rows.append(
            [
                latex_escape(name),
                fmt_num(r["n_samples"], digits=0),
                fmt_num(r["f1_macro"]),
                fmt_num(r["f1_binary"]),
                fmt_num(r["accuracy"]),
            ]
        )
    write_tabular(
        out,
        ["Dataset", "n", "F1-macro", "F1-binario", "Accuracy"],
        rows,
        col_spec="lcccc",
    )


def gen_fp_jerga():
    path = REPORTS_METRICS / "interpretability_fp_jerga.csv"
    out = GENERATED_DIR / "tab_fp_jerga.tex"
    if not path.exists() or path.stat().st_size < 5:
        placeholder(out)
        return
    df = pd.read_csv(path, index_col=0)
    series = df.iloc[:, 0]
    rows = [[latex_escape(idx), fmt_num(val, digits=0)] for idx, val in series.items()]
    write_tabular(out, ["Grupo", "FP"], rows)


def gen_jaccard():
    path = REPORTS_METRICS / "lime_shap_jaccard.csv"
    out = GENERATED_DIR / "tab_lime_shap_jaccard.tex"
    if not path.exists() or path.stat().st_size < 5:
        placeholder(out)
        return
    df = pd.read_csv(path)
    rows = [[fmt_num(r["instance"], digits=0), fmt_num(r["jaccard"])] for _, r in df.iterrows()]
    write_tabular(out, ["Instancia", "Jaccard"], rows)


def gen_lime_weights():
    path = REPORTS_METRICS / "lime_global_weights.csv"
    out = GENERATED_DIR / "tab_lime_weights.tex"
    if not path.exists() or path.stat().st_size < 10:
        placeholder(out, "LIME global weights no disponibles en esta corrida")
        return
    df = pd.read_csv(path)
    if df.empty or "class" not in df.columns:
        placeholder(out, "LIME global weights vacío")
        return
    rows = []
    for cls, group in df.groupby("class"):
        top = group.nlargest(5, "importance")
        for _, r in top.iterrows():
            rows.append(
                [latex_escape(cls), latex_escape(r["token"]), fmt_num(r["importance"])]
            )
    write_tabular(out, ["Clase", "Token", "Importancia"], rows)


def gen_val_metrics_note():
    """Opcional: tabla corta de métricas de validación desde JSON."""
    path = MODELS_DIR / "mbert-sv" / "train_metrics.json"
    out = GENERATED_DIR / "tab_val_metrics.tex"
    if not path.exists():
        placeholder(out)
        return
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)
    keys = [
        ("eval_f1_macro", "F1-macro (val)"),
        ("eval_accuracy", "Accuracy (val)"),
        ("eval_loss", "Loss (val)"),
        ("epoch", "Época"),
    ]
    rows = [[label, fmt_num(metrics.get(key))] for key, label in keys if key in metrics]
    write_tabular(out, ["Métrica", "Valor"], rows)


def copy_figures():
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    if not REPORTS_FIGURES.exists():
        return
    for png in REPORTS_FIGURES.glob("*.png"):
        dest = LATEX_FIGURES / png.name
        shutil.copy2(png, dest)
        print(f"Figura copiada: {png.name}")


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)

    gen_trivial_baselines()
    gen_class_weights()
    gen_baselines()
    gen_model_comparison()
    gen_mbert_tables()
    gen_slices()
    gen_mcnemar()
    gen_dialect_bias()
    gen_dialect_bias_prose()
    gen_dialect_bias_conclusions()
    gen_benchmark()
    gen_fp_jerga()
    gen_jaccard()
    gen_lime_weights()
    gen_val_metrics_note()
    copy_figures()
    print("Tablas LaTeX generadas en", GENERATED_DIR)


if __name__ == "__main__":
    main()
