#!/usr/bin/env python
"""Genera tablas LaTeX en reports/latex/generated/ a partir de reports/metrics/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.report_consistency import (
    assert_adversarial_consistency,
    assert_mbert_official_consistency,
    assert_mcnemar_consistency,
    assert_platform_consistency,
    assert_slice_consistency,
)
from discurso_odio.evaluation.slices import format_class_distribution, platform_rows
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
    if not {"clase", "peso_calculado", "peso_informe"}.issubset(df.columns):
        placeholder(out, "class_weights.csv sin columnas esperadas")
        return
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


def _mbert_metrics_row() -> pd.Series | None:
    path = REPORTS_METRICS / "mbert_test_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).iloc[0]


def _latex_num(value, digits: int = 4) -> str:
    """Número para prosa LaTeX con coma decimal."""
    return fmt_num(value, digits=digits).replace(".", "{,}")


def gen_mbert_official_prose():
    """Párrafo global de evaluación mBERT (cap. 06)."""
    out = GENERATED_DIR / "mbert_official_prose.tex"
    row = _mbert_metrics_row()
    if row is None:
        out.write_text("% mbert_test_metrics.csv no disponible\n", encoding="utf-8")
        return
    text = (
        f"El F1-macro es \\textbf{{{_latex_num(row['f1_macro'])}}}, con intervalo bootstrap "
        f"95\\,\\% $[{_latex_num(row['f1_bootstrap_lo'])},\\ {_latex_num(row['f1_bootstrap_hi'])}]$. "
        f"La accuracy es ${_latex_num(row['accuracy'])}$ y el MCC ${_latex_num(row['mcc'])}$. "
        f"El QWK de ${_latex_num(row['qwk'])}$ indica buen acuerdo ordinal. "
        f"El ECE de ${_latex_num(row['ece'])}$ sugiere margen de mejora en calibración."
    )
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_mbert_intro_prose():
    """Resumen ejecutivo de mBERT para la introducción."""
    out = GENERATED_DIR / "mbert_intro_prose.tex"
    row = _mbert_metrics_row()
    if row is None:
        out.write_text("% mbert_test_metrics.csv no disponible\n", encoding="utf-8")
        return
    baselines_path = REPORTS_METRICS / "baselines_tfidf.csv"
    linear_f1 = "---"
    if baselines_path.exists():
        bl = pd.read_csv(baselines_path)
        svc = bl[bl["model"].astype(str).str.lower() == "linearsvc"]
        if not svc.empty:
            linear_f1 = _latex_num(svc.iloc[0]["f1_macro"])
    text = (
        f"Sobre el conjunto de prueba ($n = 460$), el modelo mBERT alcanza un "
        f"\\textbf{{F1-macro de {_latex_num(row['f1_macro'])}}} "
        f"(intervalo bootstrap 95\\,\\%: $[{_latex_num(row['f1_bootstrap_lo'])},\\ "
        f"{_latex_num(row['f1_bootstrap_hi'])}]$), superando al mejor baseline TF-IDF "
        f"(LinearSVC, F1-macro ${linear_f1}$). El objetivo de $0{{,}}80$ no se alcanza, "
        f"aunque el intervalo de confianza superior lo incluye."
    )
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_mbert_conclusions_prose():
    """Ítem de conclusiones con F1 oficial."""
    out = GENERATED_DIR / "mbert_conclusions_prose.tex"
    row = _mbert_metrics_row()
    if row is None:
        out.write_text("% mbert_test_metrics.csv no disponible\n", encoding="utf-8")
        return
    baselines_path = REPORTS_METRICS / "baselines_tfidf.csv"
    linear_f1 = _latex_num(0.7645)
    if baselines_path.exists():
        bl = pd.read_csv(baselines_path)
        svc = bl[bl["model"].astype(str).str.lower() == "linearsvc"]
        if not svc.empty:
            linear_f1 = _latex_num(svc.iloc[0]["f1_macro"])
    line = (
        f"  \\item mBERT alcanza F1-macro \\textbf{{{_latex_num(row['f1_macro'])}}} en test "
        f"($n=460$), superando a LinearSVC (${linear_f1}$) y al resto de baselines, pero "
        f"\\textbf{{sin alcanzar}} la meta de $0{{,}}80$ definida en la configuración del "
        f"proyecto. El intervalo bootstrap superior "
        f"(${_latex_num(row['f1_bootstrap_hi'])}$) incluye el objetivo, lo que sugiere que "
        f"con más datos o calibración podría lograrse."
    )
    out.write_text(line + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_mbert_benchmark_prose():
    """Referencia al F1 local en benchmark externo."""
    out = GENERATED_DIR / "mbert_benchmark_prose.tex"
    row = _mbert_metrics_row()
    if row is None:
        out.write_text("% mbert_test_metrics.csv no disponible\n", encoding="utf-8")
        return
    text = (
        f"desempeño multiclase permanece lejos del F1-macro local de "
        f"${_latex_num(row['f1_macro'])}$"
    )
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_mbert_comparison_prose():
    """Párrafo de comparación descriptiva en cap. 06."""
    out = GENERATED_DIR / "mbert_comparison_prose.tex"
    row = _mbert_metrics_row()
    if row is None:
        out.write_text("% mbert_test_metrics.csv no disponible\n", encoding="utf-8")
        return
    baselines_path = REPORTS_METRICS / "baselines_tfidf.csv"
    linear_f1 = _latex_num(0.7645)
    if baselines_path.exists():
        bl = pd.read_csv(baselines_path)
        svc = bl[bl["model"].astype(str).str.lower() == "linearsvc"]
        if not svc.empty:
            linear_f1 = _latex_num(svc.iloc[0]["f1_macro"])
    text = (
        f"Entre baselines TF-IDF, mBERT obtiene el mejor F1-macro puntual "
        f"(${_latex_num(row['f1_macro'])}$), con LinearSVC a poca distancia "
        f"(${linear_f1}$)."
    )
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_mbert_perclass_prose():
    """Desglose por clase para cap. 06."""
    out = GENERATED_DIR / "mbert_perclass_prose.tex"
    row = _mbert_metrics_row()
    if row is None:
        out.write_text("% mbert_test_metrics.csv no disponible\n", encoding="utf-8")
        return
    classes = [
        ("Lenguaje Ofensivo", "f1_Lenguaje Ofensivo"),
        ("No Tóxico", "f1_No Tóxico"),
        ("Discurso de Odio", "f1_Discurso de Odio"),
    ]
    ranked = sorted(
        ((name, float(row[col])) for name, col in classes if col in row.index),
        key=lambda item: item[1],
        reverse=True,
    )
    best_name, best_f1 = ranked[0]
    second_name, second_f1 = ranked[1]
    weakest_name, weakest_f1 = ranked[-1]
    text = (
        f"{latex_escape(best_name)} obtiene el mejor F1 (${_latex_num(best_f1)}$), seguido de "
        f"{latex_escape(second_name)} (${_latex_num(second_f1)}$). "
        f"{latex_escape(weakest_name)} es la clase más débil (${_latex_num(weakest_f1)}$), "
        f"aunque el soporte es menor."
    )
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_mbert_traceability():
    """Tabla de trazabilidad F1-macro histórico vs oficial."""
    path = REPORTS_METRICS / "mbert_f1_traceability.csv"
    out = GENERATED_DIR / "tab_mbert_traceability.tex"
    if not path.exists():
        placeholder(out, "mbert_f1_traceability.csv no disponible")
        return
    df = pd.read_csv(path)
    status_labels = {
        "official": "Oficial",
        "same_run_rounded": "Mismo run (redondeo)",
        "validation_only": "Solo validación",
        "not_reproducible": "No reproducible",
    }
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                fmt_num(r["f1_macro_published"]),
                latex_escape(status_labels.get(str(r["status"]), str(r["status"]))),
                latex_escape(str(r.get("source_file", ""))),
                latex_escape(str(r.get("checkpoint", ""))),
                latex_escape(str(r.get("split", ""))),
            ]
        )
    write_tabular(
        out,
        ["F1-macro", "Estado", "Fuente", "Checkpoint", "Split"],
        rows,
        col_spec="lclll",
    )


def gen_mbert_traceability_prose():
    """Párrafo que explica la tabla de trazabilidad."""
    out = GENERATED_DIR / "mbert_traceability_prose.tex"
    run_path = REPORTS_METRICS / "mbert_official_run.json"
    if not run_path.exists():
        out.write_text("% mbert_official_run.json no disponible\n", encoding="utf-8")
        return
    with open(run_path, encoding="utf-8") as handle:
        run = json.load(handle)
    text = (
        f"La Tabla~\\ref{{tab:mbert-traceability}} documenta cada cifra histórica de "
        f"F1-macro citada para mBERT. La métrica \\textbf{{oficial}} "
        f"(${_latex_num(run['official_f1_macro'])}$) proviene del checkpoint "
        f"\\texttt{{{latex_escape(run['checkpoint'])}}} "
        f"(semilla {int(run['seed'])}, hash de test "
        f"\\texttt{{{latex_escape(str(run['test_sha256'])[:16])}}}\\ldots, "
        f"evaluación \\texttt{{{latex_escape(str(run['evaluated_at'])[:10])}}}). "
        f"La cifra $0{{,}}7862$ del informe de auditoría queda archivada como "
        f"\\emph{{no reproducible}} y no se usa en comparaciones."
    )
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


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


def _model_label(model: str) -> str:
    labels = {"mbert": "mBERT", "xlm-roberta": "XLM-RoBERTa"}
    return labels.get(str(model), str(model))


def _platform_label(slice_name: str) -> str:
    labels = {"plataforma_X": "X", "plataforma_Facebook": "Facebook"}
    return labels.get(slice_name, slice_name.replace("plataforma_", ""))


def _slice_rows_for_report(df: pd.DataFrame, model: str | None = "mbert") -> pd.DataFrame:
    if df.empty:
        return df
    if "model" in df.columns and model and model in df["model"].values:
        return df[df["model"] == model].copy()
    if "model" in df.columns:
        return df[df["model"] == df["model"].iloc[0]].copy()
    return df.copy()


def _slice_value(df: pd.DataFrame, slice_name: str, column: str = "f1_macro") -> float | None:
    matches = df[df["slice"] == slice_name]
    if matches.empty:
        return None
    value = matches.iloc[0][column]
    if pd.isna(value):
        return None
    return float(value)


def _short_checkpoint(value: str) -> str:
    text = str(value)
    if "checkpoint-" in text:
        idx = text.find("checkpoint-")
        return text[idx:]
    if len(text) > 40:
        return Path(text).name
    return text


def _format_ci_or_warning(row: pd.Series) -> str:
    low_power = bool(row.get("low_power", False))
    if low_power:
        note = str(row.get("power_note", ""))
        if note:
            return latex_escape(note)
        return f"baja potencia ($n={int(row['n'])}$)"
    if pd.notna(row.get("f1_ci_lo")) and pd.notna(row.get("f1_ci_hi")):
        return f"[{fmt_num(row['f1_ci_lo'])}, {fmt_num(row['f1_ci_hi'])}]"
    return "---"


def _format_distribution(row: pd.Series) -> str:
    if any(f"n{i}" in row.index for i in range(4)):
        return format_class_distribution(row)
    return "---"


def gen_slices():
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = GENERATED_DIR / "tab_slices.tex"
    if not path.exists():
        placeholder(out)
        return
    raw = pd.read_csv(path)
    multi_model = "model" in raw.columns and raw["model"].nunique() > 1
    df = _slice_rows_for_report(raw)
    rows = []
    for _, r in df.iterrows():
        definition = latex_escape(r.get("definition", r["slice"]))
        row = [
            latex_escape(r["slice"]),
            definition,
            fmt_num(r.get("n"), digits=0),
            _format_distribution(r),
            fmt_num(r["f1_macro"]),
            _format_ci_or_warning(r),
            fmt_num(r["accuracy"]),
        ]
        if multi_model:
            row = [latex_escape(_model_label(r.get("model", "")))] + row
        rows.append(row)
    headers = ["Corte", "Definición", "n", "Clases", "F1-macro", "IC / nota", "Accuracy"]
    col_spec = "llrrccc"
    if multi_model:
        headers = ["Modelo"] + headers
        col_spec = "lllrrccc"
    write_tabular(out, headers, rows, col_spec=col_spec)


def gen_slice_perclass_table():
    path = REPORTS_METRICS / "slice_perclass_metrics.csv"
    out = GENERATED_DIR / "tab_slice_perclass.tex"
    if not path.exists():
        out.write_text("% slice_perclass_metrics.csv no disponible\n", encoding="utf-8")
        return
    df = pd.read_csv(path)
    if df.empty:
        placeholder(out, "Sin métricas por clase en cortes")
        return
    mbert = df[df["model"] == "mbert"] if "model" in df.columns else df
    if mbert.empty:
        placeholder(out, "Sin métricas mBERT por clase en cortes")
        return
    rows = []
    for _, r in mbert.sort_values(["slice", "class_label"]).iterrows():
        rows.append(
            [
                latex_escape(r["slice"]),
                latex_escape(r.get("class_name", r.get("class_label", ""))),
                fmt_num(r.get("support"), digits=0),
                fmt_num(r.get("precision")),
                fmt_num(r.get("recall")),
                fmt_num(r.get("f1")),
            ]
        )
    write_tabular(
        out,
        ["Corte", "Clase", "Support", "Precision", "Recall", "F1"],
        rows,
        col_spec="llrrrr",
    )


def gen_platform_table():
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = GENERATED_DIR / "tab_platform.tex"
    if not path.exists():
        placeholder(out, "slice_metrics.csv no disponible")
        return
    df = platform_rows(pd.read_csv(path))
    if df.empty:
        placeholder(out, "Sin métricas por plataforma")
        return
    rows = []
    for _, r in df.sort_values(["model", "slice"]).iterrows():
        rows.append(
            [
                latex_escape(_model_label(r["model"])),
                latex_escape(_platform_label(r["slice"])),
                latex_escape(_short_checkpoint(str(r.get("checkpoint", "")))),
                fmt_num(r.get("n"), digits=0),
                _format_distribution(r),
                fmt_num(r["f1_macro"]),
                fmt_num(r["accuracy"]),
                _format_ci_or_warning(r),
            ]
        )
    write_tabular(
        out,
        ["Modelo", "Plataforma", "Checkpoint", "n", "Clases", "F1-macro", "Accuracy", "IC / nota"],
        rows,
        col_spec="lllrrrrc",
    )


def _platform_prose_lines(df: pd.DataFrame, model: str) -> list[str]:
    sub = df[df["model"] == model] if "model" in df.columns else df
    sub = platform_rows(sub)
    if sub.empty:
        return []
    lines = ["\\begin{itemize}[leftmargin=*]"]
    for _, r in sub.sort_values("slice").iterrows():
        plat = _platform_label(r["slice"])
        dist = _format_distribution(r)
        ci_note = _format_ci_or_warning(r)
        ckpt = latex_escape(_short_checkpoint(str(r.get("checkpoint", ""))))
        lines.append(
            f"  \\item \\textbf{{{plat}}} ($n={int(r['n'])}$, clases {dist}): F1-macro "
            f"\\textbf{{{fmt_num(r['f1_macro'])}}}, accuracy {fmt_num(r['accuracy'])} "
            f"({ci_note}; checkpoint \\texttt{{{ckpt}}})."
        )
    lines.append(
        "  \\item Fuente: \\texttt{reports/metrics/slice\\_metrics.csv}."
    )
    lines.append("\\end{itemize}")
    return lines


def gen_platform_prose_mbert():
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = GENERATED_DIR / "platform_prose_mbert.tex"
    if not path.exists():
        out.write_text("% slice_metrics.csv no disponible\n", encoding="utf-8")
        return
    df = pd.read_csv(path)
    if "mbert" not in df.get("model", pd.Series(dtype=str)).values:
        out.write_text("% métricas mBERT por plataforma no disponibles\n", encoding="utf-8")
        return
    lines = _platform_prose_lines(df, "mbert")
    run_path = REPORTS_METRICS / "mbert_official_run.json"
    if run_path.exists():
        with open(run_path, encoding="utf-8") as handle:
            run = json.load(handle)
        lines.insert(
            1,
            f"  \\item Corrida oficial: \\texttt{{run\\_id={latex_escape(run.get('run_id', ''))}}}, "
            f"checkpoint \\texttt{{{latex_escape(run.get('checkpoint', 'checkpoint-1072'))}}}, "
            f"evaluado {latex_escape(str(run.get('evaluated_at', '')))}.",
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_platform_prose_xlmr():
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = GENERATED_DIR / "platform_prose_xlmr.tex"
    if not path.exists():
        out.write_text(
            "% slice_metrics.csv no disponible; "
            "% predicciones XLM-R en reports/metrics/xlmr_test_predictions.csv no cargadas\n",
            encoding="utf-8",
        )
        return
    df = pd.read_csv(path)
    if "xlm-roberta" not in df.get("model", pd.Series(dtype=str)).values:
        out.write_text(
            "% Métricas XLM-R por plataforma no disponibles: "
            "% falta reports/metrics/xlmr_test_predictions.csv o no se ejecutó compute_slice_metrics.py. "
            "% No se transcriben cifras de mBERT.\n",
            encoding="utf-8",
        )
        return
    lines = _platform_prose_lines(df, "xlm-roberta")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_platform_traceability():
    path = REPORTS_METRICS / "mbert_platform_traceability.csv"
    out_tab = GENERATED_DIR / "tab_mbert_platform_traceability.tex"
    out_prose = GENERATED_DIR / "mbert_platform_traceability_prose.tex"
    if not path.exists():
        out_tab.write_text("% mbert_platform_traceability.csv no disponible\n", encoding="utf-8")
        out_prose.write_text("% mbert_platform_traceability.csv no disponible\n", encoding="utf-8")
        return
    df = pd.read_csv(path)
    if df.empty:
        placeholder(out_tab, "Sin trazabilidad de plataforma")
        return
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                latex_escape(r.get("platform", "")),
                fmt_num(r.get("f1_macro_published")),
                fmt_num(r.get("n"), digits=0) if pd.notna(r.get("n")) else "---",
                latex_escape(r.get("status", "")),
                latex_escape(r.get("checkpoint", "")),
                latex_escape(str(r.get("source_file", ""))),
            ]
        )
    write_tabular(
        out_tab,
        ["Plataforma", "F1-macro", "n", "Estado", "Checkpoint", "Fuente"],
        rows,
        col_spec="lrrlll",
    )
    official = df[df["status"] == "official"]
    historical = df[df["status"] == "not_reproducible"]
    prose_lines = [
        "Las cifras oficiales por plataforma proceden de la misma evaluación T-04/T-05 "
        "sobre \\texttt{mbert\\_test\\_predictions.csv}.",
    ]
    if not official.empty:
        for _, r in official.iterrows():
            prose_lines.append(
                f"Oficial {latex_escape(r['platform'])}: F1-macro "
                f"\\textbf{{{fmt_num(r['f1_macro_published'])}}} ($n={int(r['n'])}$)."
            )
    if not historical.empty:
        hist_parts = [
            f"{latex_escape(r['platform'])} ({fmt_num(r['f1_macro_published'])})"
            for _, r in historical.iterrows()
        ]
        prose_lines.append(
            "Cifras históricas del informe de Cristian ("
            + ", ".join(hist_parts)
            + ") se marcan como \\texttt{not\\_reproducible} y no se mezclan con la tabla oficial."
        )
    out_prose.write_text("\n".join(prose_lines) + "\n", encoding="utf-8")
    print(f"Generado: {out_prose.name}")


def gen_slices_prose():
    """Genera párrafo LaTeX con cifras de cortes derivadas del artefacto."""
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = GENERATED_DIR / "slices_prose.tex"
    if not path.exists():
        out.write_text("% slice_metrics.csv no disponible\n", encoding="utf-8")
        return

    df = _slice_rows_for_report(pd.read_csv(path))
    if df.empty:
        out.write_text("% slice_metrics.csv vacío\n", encoding="utf-8")
        return

    f1_x = _slice_value(df, "plataforma_X")
    f1_fb = _slice_value(df, "plataforma_Facebook")
    f1_ironia_si = _slice_value(df, "ironia_si")
    f1_ironia_no = _slice_value(df, "ironia_no")
    f1_sarcasmo_si = _slice_value(df, "sarcasmo_si")
    f1_sarcasmo_no = _slice_value(df, "sarcasmo_no")
    f1_pol_si = _slice_value(df, "contexto_politico_si")
    f1_pol_no = _slice_value(df, "contexto_politico_no")

    lines = [
        "\\begin{itemize}[leftmargin=*]",
    ]
    if f1_x is not None and f1_fb is not None:
        lines.append(
            f"  \\item \\textbf{{Plataforma X}} (F1 \\textbf{{{fmt_num(f1_x)}}}) vs.\\ "
            f"\\textbf{{Facebook}} (F1 \\textbf{{{fmt_num(f1_fb)}}}): brecha importante."
        )
    if f1_ironia_si is not None and f1_ironia_no is not None:
        ironia_row = df[df["slice"] == "ironia_si"]
        ironia_note = ""
        if not ironia_row.empty and bool(ironia_row.iloc[0].get("low_power", False)):
            ironia_note = f" ({_format_ci_or_warning(ironia_row.iloc[0])})"
        lines.append(
            f"  \\item \\textbf{{Ironía}} (F1 \\textbf{{{fmt_num(f1_ironia_si)}}}{ironia_note}) y "
            f"\\textbf{{sarcasmo}} (F1 \\textbf{{{fmt_num(f1_sarcasmo_si)}}}) "
            f"degradan el desempeño respecto a sus contrapartes sin figura retórica "
            f"(F1 {fmt_num(f1_ironia_no)} e {fmt_num(f1_sarcasmo_no)})."
        )
    elif f1_ironia_si is not None:
        lines.append(
            f"  \\item \\textbf{{Ironía}} (F1 \\textbf{{{fmt_num(f1_ironia_si)}}}) "
            f"degrada el desempeño respecto al subconjunto sin ironía."
        )
    if f1_pol_si is not None and f1_pol_no is not None:
        if f1_pol_si >= f1_pol_no:
            lines.append(
                "  \\item Contexto político no perjudica; incluso mejora ligeramente "
                f"(F1 {fmt_num(f1_pol_si)} vs.\\ {fmt_num(f1_pol_no)})."
            )
        else:
            lines.append(
                "  \\item Contexto político muestra peor desempeño "
                f"(F1 {fmt_num(f1_pol_si)} vs.\\ {fmt_num(f1_pol_no)})."
            )
    lines.append(
        "  \\item Fuente: \\texttt{reports/metrics/slice\\_metrics.csv}."
    )
    lines.append("\\end{itemize}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_slices_intro():
    """Snippet breve para la introducción con los cortes más difíciles."""
    path = REPORTS_METRICS / "slice_metrics.csv"
    out = GENERATED_DIR / "slices_intro.tex"
    if not path.exists():
        out.write_text("% slice_metrics.csv no disponible\n", encoding="utf-8")
        return

    df = _slice_rows_for_report(pd.read_csv(path))
    f1_x = _slice_value(df, "plataforma_X")
    f1_ironia = _slice_value(df, "ironia_si")
    f1_sarcasmo = _slice_value(df, "sarcasmo_si")

    parts = []
    if f1_ironia is not None and f1_sarcasmo is not None:
        parts.append(
            f"Los cortes más difíciles son ironía (F1 $\\approx$ {fmt_num(f1_ironia, digits=2)}) "
            f"y sarcasmo (F1 $\\approx$ {fmt_num(f1_sarcasmo, digits=2)})"
        )
    if f1_x is not None:
        suffix = (
            f", mientras que el desempeño en la plataforma X es sólido "
            f"(F1 $\\approx$ {fmt_num(f1_x, digits=2)})"
        )
        parts[-1] = parts[-1] + suffix if parts else (
            f"El desempeño en la plataforma X es sólido (F1 $\\approx$ {fmt_num(f1_x, digits=2)})"
        )
    if parts:
        out.write_text(parts[0] + ".\n", encoding="utf-8")
    else:
        out.write_text("% slices sin datos suficientes\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def _mcnemar_model_label(name: str) -> str:
    labels = {
        "mbert": "mBERT",
        "xlm-roberta": "XLM-RoBERTa",
        "logisticregression": "LogisticRegression",
        "linearsvc": "LinearSVC",
        "multinomialnb": "MultinomialNB",
    }
    return labels.get(str(name).lower(), str(name))


def gen_mcnemar():
    path = REPORTS_METRICS / "mcnemar_tests.csv"
    out = GENERATED_DIR / "tab_mcnemar.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        if "model_b" in df.columns:
            model_b = _mcnemar_model_label(r["model_b"])
        else:
            model_b = latex_escape(r.get("baseline", "---"))
        sig = "Sí" if bool(r.get("significant", False)) else "No"
        rows.append(
            [
                _mcnemar_model_label(r.get("model_a", "mbert")),
                model_b,
                fmt_num(r["statistic"]),
                fmt_num(r["p_value"], digits=6),
                fmt_num(r["b"], digits=0),
                fmt_num(r["c"], digits=0),
                fmt_num(r.get("n", 460), digits=0),
                sig,
            ]
        )
    write_tabular(
        out,
        ["Modelo A", "Modelo B", "Estadístico", "p-valor", "b", "c", "n", "Sig."],
        rows,
        col_spec="llcccccc",
    )


def gen_mcnemar_prose():
    path = REPORTS_METRICS / "mcnemar_tests.csv"
    out = GENERATED_DIR / "mcnemar_prose.tex"
    if not path.exists():
        out.write_text("% mcnemar_tests.csv no disponible\n", encoding="utf-8")
        print(f"Generado: {out.name}")
        return

    df = pd.read_csv(path)
    if "model_b" not in df.columns:
        out.write_text("% esquema legacy de mcnemar_tests.csv\n", encoding="utf-8")
        print(f"Generado: {out.name}")
        return

    xlmr = df[df["model_b"] == "xlm-roberta"]
    lines: list[str] = []

    if not xlmr.empty:
        row = xlmr.iloc[0]
        p_fmt = fmt_num(row["p_value"], digits=6)
        b = int(row["b"])
        c = int(row["c"])
        n = int(row.get("n", 460))
        alpha = float(row.get("alpha", 0.05))
        if bool(row.get("significant", False)):
            lines.append(
                f"La comparación pareada mBERT vs.\\ XLM-RoBERTa sobre las mismas "
                f"$n={n}$ instancias de test arroja $p={p_fmt}$ "
                f"($b={b}$, $c={c}$; $\\alpha={alpha:.2f}$): la diferencia "
                f"\\textbf{{es estadísticamente significativa}} bajo el umbral definido."
            )
        else:
            lines.append(
                f"La comparación pareada mBERT vs.\\ XLM-RoBERTa sobre las mismas "
                f"$n={n}$ instancias de test arroja $p={p_fmt}$ "
                f"($b={b}$, $c={c}$; $\\alpha={alpha:.2f}$): "
                f"\\textbf{{no se rechaza la hipótesis nula}}; la diferencia puntual "
                f"de F1-macro no es concluyente con este tamaño de muestra "
                f"(intervalos bootstrap solapados)."
            )

    baselines = df[df["model_b"].isin(["logisticregression", "linearsvc", "multinomialnb"])]
    sig_baselines = baselines[baselines.get("significant", False) == True]  # noqa: E712
    if not sig_baselines.empty:
        names = ", ".join(_mcnemar_model_label(b) for b in sig_baselines["model_b"])
        lines.append(
            f"Frente a {names} la superioridad de mBERT sí alcanza significancia "
            f"estadística ($\\alpha={float(baselines.iloc[0].get('alpha', 0.05)):.2f}$)."
        )

    if lines:
        out.write_text(" ".join(lines) + "\n", encoding="utf-8")
    else:
        out.write_text("% McNemar sin filas interpretables\n", encoding="utf-8")
    print(f"Generado: {out.name}")


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


def _load_adversarial_summary() -> dict | None:
    path = REPORTS_METRICS / "mbert_adversarial_summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def gen_adversarial_families():
    path = REPORTS_METRICS / "mbert_adversarial_family_metrics.csv"
    out = GENERATED_DIR / "tab_adversarial_families.tex"
    if not path.exists():
        placeholder(out, "mbert_adversarial_family_metrics.csv no disponible")
        return
    df = pd.read_csv(path)
    if df.empty:
        placeholder(out)
        return
    sub = df[df["model"] == "mbert"] if "mbert" in df["model"].values else df
    rows = []
    for _, r in sub.iterrows():
        key = str(r.get("family_key", ""))
        if r.get("metric_type") == "fpr":
            metric = f"FPR {_frac_latex(r.get('fp', 0), r.get('n', 0))}"
        else:
            metric = f"{int(r.get('correct', 0))}/{int(r.get('n', 0))}"
        rows.append(
            [
                latex_escape(key),
                latex_escape(str(r.get("grupo", ""))[:40]),
                metric,
                fmt_num(r.get("conf_mean")),
                fmt_num(r.get("n"), digits=0),
            ]
        )
    write_tabular(
        out,
        ["Fam.", "Grupo", "Resultado", "Conf.", "n"],
        rows,
        col_spec="llccc",
    )


def gen_adversarial_dialect():
    path = REPORTS_METRICS / "mbert_adversarial_family_metrics.csv"
    out = GENERATED_DIR / "tab_adversarial_dialect.tex"
    if not path.exists():
        placeholder(out)
        return
    df = pd.read_csv(path)
    sub = df[df["model"] == "mbert"] if "mbert" in df["model"].values else df
    rows = []
    for key in ("D", "E"):
        part = sub[sub["family_key"] == key]
        if part.empty:
            continue
        r = part.iloc[0]
        rows.append(
            [
                "Jerga SV" if key == "D" else "Español estándar",
                _frac_latex(r.get("fp", 0), r.get("n", 0)),
                fmt_num(r.get("fpr")),
                fmt_num(r.get("fpr_ci_lo")),
                fmt_num(r.get("fpr_ci_hi")),
                fmt_num(r.get("n"), digits=0),
            ]
        )
    if not rows:
        placeholder(out, "Grupos D/E no disponibles")
        return
    write_tabular(
        out,
        ["Grupo", "FP/n", "FPR", "IC lo", "IC hi", "n"],
        rows,
        col_spec="lccccr",
    )


def gen_adversarial_prose():
    out = GENERATED_DIR / "adversarial_prose.tex"
    summary = _load_adversarial_summary()
    if not summary:
        out.write_text("% mbert_adversarial_summary.json no disponible\n", encoding="utf-8")
        return
    lines = [
        f"Se evaluó el checkpoint \\texttt{{{latex_escape(summary.get('checkpoint', ''))}}} "
        f"(\\texttt{{run\\_id}}={latex_escape(str(summary.get('run_id', ''))[:8])}\\ldots) "
        f"sobre la batería adversarial \\texttt{{{latex_escape(summary.get('battery_version', ''))}}} "
        f"($n={summary.get('n_cases', 75)}$, SHA-256 verificado). "
        f"Esta métrica es \\textbf{{diagnóstica}} y no sustituye el F1-macro oficial sobre "
        f"\\texttt{{test.csv}}.",
        f"El modelo acertó {summary.get('global_correct')}/{summary.get('n_cases')} casos "
        f"(F1-macro diagnóstico={fmt_num(summary.get('f1_macro_diagnostic'))}). "
        f"Sobre amenazas ofuscadas (grupo C): F1 Amenazas/Violencia="
        f"{fmt_num(summary.get('f1_amenazas'))}, FN={summary.get('threat_fn')}/"
        f"{summary.get('n_threats')}. "
        f"La brecha FPR dialectal (D$-$E) es {fmt_num(summary.get('fpr_dialectal_gap'))}.",
    ]
    if summary.get("mitigation_required"):
        reasons = ", ".join(summary.get("mitigation_reasons", []))
        lines.append(
            f"La puerta de mitigación se activó ({latex_escape(reasons)}); "
            f"se entrenó un candidato B2 sin filas de train que coinciden con la batería."
        )
    else:
        lines.append(
            "No se activó la puerta de mitigación; se documentan riesgos residuales "
            "(tamaño reducido de la batería y anchos IC en subconjuntos pequeños)."
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_adversarial_before_after():
    path = REPORTS_METRICS / "mbert_adversarial_before_after.csv"
    out = GENERATED_DIR / "tab_adversarial_before_after.tex"
    if not path.exists():
        placeholder(out, "Comparación antes/después no disponible")
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                latex_escape(str(r.get("stage", ""))),
                latex_escape(str(r.get("checkpoint", ""))),
                fmt_num(r.get("test_f1_macro")),
                f"{int(r.get('group_c_correct', 0))}/11",
                f"{int(r.get('group_b_correct', 0))}/18",
                fmt_num(r.get("fpr_gap")),
                fmt_num(r.get("adv_global_correct"), digits=0) + "/" + fmt_num(r.get("adv_n"), digits=0),
            ]
        )
    write_tabular(
        out,
        ["Etapa", "Checkpoint", "F1 test", "C", "B", "FPR gap", "Adv."],
        rows,
        col_spec="llccccr",
    )


def gen_adversarial_tradeoffs():
    out = GENERATED_DIR / "adversarial_tradeoffs.tex"
    meta_path = REPORTS_METRICS / "mbert_adversarial_before_after.json"
    if not meta_path.exists():
        out.write_text(
            "% Sin candidato B2: riesgos residuales documentados en adversarial_prose.tex\n",
            encoding="utf-8",
        )
        return
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    adopted = meta.get("mitigation_adopted")
    notes = meta.get("adoption_notes", "")
    if adopted:
        line = (
            f"El candidato B2 cumple los criterios de adopción documentados en T-06: "
            f"{latex_escape(notes)}. El checkpoint oficial T-04 permanece como referencia "
            f"de la métrica publicada."
        )
    else:
        line = (
            f"El candidato B2 no se adoptó como mitigación oficial: {latex_escape(notes)}. "
            f"Se mantienen los riesgos residuales en amenazas ofuscadas, emojis y jerga dialectal."
        )
    out.write_text(line + "\n", encoding="utf-8")
    print(f"Generado: {out.name}")


def gen_adversarial_conclusions():
    out = GENERATED_DIR / "adversarial_conclusions.tex"
    summary = _load_adversarial_summary()
    if not summary:
        out.write_text("% adversarial summary no disponible\n", encoding="utf-8")
        return
    line = (
        f"  \\item La batería adversarial ({summary.get('global_correct')}/{summary.get('n_cases')} "
        f"aciertos globales) confirma brechas en robustez retórica y equidad dialectal del "
        f"checkpoint oficial; ver Tabla~\\ref{{tab:adversarial-families}}."
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
    note_path = REPORTS_METRICS / "interpretability_note.json"
    if note_path.exists():
        note = pd.read_json(note_path, typ="series")
        if not bool(note.get("jaccard_available", True)):
            placeholder(
                out,
                "Jaccard LIME--SHAP omitido: solapamiento nulo tras alineación WordPiece (ver nota en capítulo)",
            )
            return
    if not path.exists() or path.stat().st_size < 5:
        placeholder(out, "Jaccard LIME--SHAP no disponible en esta corrida")
        return
    df = pd.read_csv(path)
    if df.empty or float(df["jaccard"].sum()) == 0.0:
        placeholder(out, "Jaccard LIME--SHAP omitido: todas las instancias con solapamiento 0")
        return
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


def gen_inference_contract_prose():
    """Genera párrafo sobre el contrato de inferencia y delta vs auditoría."""
    contract_path = REPORTS_METRICS / "inference_contract.json"
    delta_path = REPORTS_METRICS / "parity_audit_delta.json"
    out = GENERATED_DIR / "inference_contract_prose.tex"

    if not contract_path.exists():
        out.write_text("% inference_contract.json no disponible\n", encoding="utf-8")
        return

    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)

    delta_block = ""
    if delta_path.exists():
        with open(delta_path, encoding="utf-8") as f:
            delta = json.load(f)
        f1_off = delta.get("official_contract", {}).get("f1_macro", "---")
        f1_256 = delta.get("diagnostic_audit_path", {}).get("f1_macro", "---")
        pct_256 = delta.get("vs_audit_256", {}).get("pct_changed", "---")
        pct_raw = delta.get("vs_raw_no_normalization", {}).get("pct_changed", "---")
        audit_f1 = delta.get("audit_reference", {}).get("f1_macro_reported", 0.8058)
        delta_block = (
            f" La auditoría reportó F1-macro {audit_f1:.4f} con evaluación a "
            f"\\texttt{{max\\_length}}=256; bajo el contrato oficial "
            f"(\\texttt{{max\\_length}}={contract['max_length']}, "
            f"normalizador {latex_escape(contract['normalizer'])} v"
            f"{latex_escape(contract['normalizer_version'])}), el F1-macro es "
            f"{f1_off} ({pct_256}\\% de predicciones distintas frente a la ruta "
            f"diagnóstica a 256, F1={f1_256}). Sin normalización sobre texto crudo, "
            f"{pct_raw}\\% de las predicciones cambian respecto al contrato."
        )

    text = (
        f"Entrenamiento, evaluación por lote e inferencia del prototipo XLM-R comparten "
        f"un contrato versionado (\\texttt{{inference\\_contract.json}}): "
        f"\\texttt{{max\\_length}}={contract['max_length']}, columna "
        f"\\texttt{{{latex_escape(contract['text_column'])}}} y normalizador "
        f"\\texttt{{{latex_escape(contract['normalizer'])}}} "
        f"v{latex_escape(contract['normalizer_version'])}."
        f"{delta_block}\n"
    )
    out.write_text(text, encoding="utf-8")
    print(f"Generado: {out.name}")


def copy_figures():
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    if not REPORTS_FIGURES.exists():
        return
    for png in REPORTS_FIGURES.glob("*.png"):
        dest = LATEX_FIGURES / png.name
        try:
            shutil.copy2(png, dest)
            print(f"Figura copiada: {png.name}")
        except OSError as exc:
            print(f"Advertencia: no se pudo copiar {png.name}: {exc}")


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)

    gen_trivial_baselines()
    gen_class_weights()
    gen_baselines()
    gen_model_comparison()
    gen_mbert_tables()
    gen_mbert_official_prose()
    gen_mbert_intro_prose()
    gen_mbert_conclusions_prose()
    gen_mbert_benchmark_prose()
    gen_mbert_comparison_prose()
    gen_mbert_perclass_prose()
    gen_mbert_traceability()
    gen_mbert_traceability_prose()
    gen_slices()
    gen_slice_perclass_table()
    gen_platform_table()
    gen_platform_prose_mbert()
    gen_platform_prose_xlmr()
    gen_platform_traceability()
    gen_slices_prose()
    gen_slices_intro()
    gen_mcnemar()
    gen_mcnemar_prose()
    gen_dialect_bias()
    gen_dialect_bias_prose()
    gen_dialect_bias_conclusions()
    gen_adversarial_families()
    gen_adversarial_dialect()
    gen_adversarial_prose()
    gen_adversarial_before_after()
    gen_adversarial_tradeoffs()
    gen_adversarial_conclusions()
    gen_benchmark()
    gen_fp_jerga()
    gen_jaccard()
    gen_lime_weights()
    gen_val_metrics_note()
    gen_inference_contract_prose()
    assert_platform_consistency(REPORTS_METRICS / "slice_metrics.csv", GENERATED_DIR)
    assert_slice_consistency(REPORTS_METRICS / "slice_metrics.csv", GENERATED_DIR)
    assert_mcnemar_consistency(REPORTS_METRICS / "mcnemar_tests.csv", GENERATED_DIR)
    assert_mbert_official_consistency(REPORTS_METRICS / "mbert_test_metrics.csv", GENERATED_DIR)
    assert_adversarial_consistency(GENERATED_DIR)
    copy_figures()
    print("Tablas LaTeX generadas en", GENERATED_DIR)


if __name__ == "__main__":
    main()
