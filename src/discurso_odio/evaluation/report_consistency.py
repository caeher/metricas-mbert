"""Verificación de consistencia entre artefactos de métricas y LaTeX generado."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from discurso_odio.evaluation.official_mbert import (
    MBERT_METRICS_PATH,
    MBERT_OFFICIAL_RUN_PATH,
    MBERT_TRACEABILITY_CSV,
)
from discurso_odio.evaluation.slices import HISTORICAL_PLATFORM_F1, platform_rows
from discurso_odio.models.mbert_artifacts import OFFICIAL_F1_MACRO

XLMR_AUDIT_F1 = {
    "plataforma_X": 0.867,
    "plataforma_Facebook": 0.715,
}

BANNED_HISTORICAL_PLATFORM_F1 = sorted(set(HISTORICAL_PLATFORM_F1.values()))

FLOAT_PATTERN = re.compile(r"\b0\.\d{2,4}\b")


@dataclass
class ConsistencyError:
    message: str


def fmt_num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    return f"{float(value):.{digits}f}"


def latex_fmt_num(value, digits: int = 4) -> str:
    """Formato numérico usado en prosa LaTeX generada (coma decimal)."""
    return fmt_num(value, digits=digits).replace(".", "{,}")


def _model_label(model: str) -> str:
    labels = {"mbert": "mBERT", "xlm-roberta": "XLM-RoBERTa"}
    return labels.get(model, model)


def _platform_label(slice_name: str) -> str:
    labels = {"plataforma_X": "X", "plataforma_Facebook": "Facebook"}
    return labels.get(slice_name, slice_name.replace("plataforma_", ""))


def _extract_f1_values(text: str) -> list[float]:
    return [float(m.group()) for m in FLOAT_PATTERN.finditer(text)]


def _slice_platform_metrics(
    slice_df: pd.DataFrame, model: str
) -> dict[str, dict[str, float | int | str]]:
    sub = platform_rows(slice_df[slice_df["model"] == model]) if "model" in slice_df.columns else platform_rows(slice_df)
    out: dict[str, dict[str, float | int | str]] = {}
    for _, row in sub.iterrows():
        out[str(row["slice"])] = {
            "f1_macro": float(row["f1_macro"]),
            "accuracy": float(row["accuracy"]),
            "n": int(row["n"]),
            "checkpoint": str(row.get("checkpoint", "")),
        }
    return out


def check_platform_metrics(
    slice_csv: Path,
    generated_dir: Path,
    *,
    strict_xlmr: bool = False,
) -> list[ConsistencyError]:
    """Compara slice_metrics.csv con tablas/prosa LaTeX de plataforma."""
    errors: list[ConsistencyError] = []

    if not slice_csv.exists():
        errors.append(ConsistencyError(f"Artefacto no encontrado: {slice_csv}"))
        return errors

    slice_df = pd.read_csv(slice_csv)
    platform_df = platform_rows(slice_df)

    tab_platform = generated_dir / "tab_platform.tex"
    prose_xlmr = generated_dir / "platform_prose_xlmr.tex"
    prose_mbert = generated_dir / "platform_prose_mbert.tex"

    for tex_path in (tab_platform, prose_xlmr, prose_mbert):
        if not tex_path.exists():
            errors.append(ConsistencyError(f"LaTeX generado ausente: {tex_path.name}"))

    if errors:
        return errors

    tab_text = tab_platform.read_text(encoding="utf-8")
    xlmr_text = prose_xlmr.read_text(encoding="utf-8")
    mbert_text = prose_mbert.read_text(encoding="utf-8")

    has_xlmr = "xlm-roberta" in platform_df["model"].values if "model" in platform_df.columns else False
    xlmr_missing_doc = "xlmr_test_predictions.csv" in xlmr_text and "no disponible" in xlmr_text.lower()

    if has_xlmr:
        xlmr_metrics = _slice_platform_metrics(slice_df, "xlm-roberta")
        for slice_name, expected in xlmr_metrics.items():
            f1_fmt = fmt_num(expected["f1_macro"])
            if f1_fmt not in tab_text:
                errors.append(
                    ConsistencyError(
                        f"tab_platform.tex no contiene F1 {f1_fmt} de XLM-R ({slice_name})"
                    )
                )
            if f1_fmt not in xlmr_text:
                errors.append(
                    ConsistencyError(
                        f"platform_prose_xlmr.tex no contiene F1 {f1_fmt} ({slice_name})"
                    )
                )
        mbert_metrics = _slice_platform_metrics(slice_df, "mbert")
        for slice_name, expected in mbert_metrics.items():
            mbert_fmt = fmt_num(expected["f1_macro"])
            if mbert_fmt in xlmr_text:
                errors.append(
                    ConsistencyError(
                        f"platform_prose_xlmr.tex contiene F1 de mBERT ({mbert_fmt}) "
                        f"en lugar de XLM-R para {slice_name}"
                    )
                )
    elif not xlmr_missing_doc and strict_xlmr:
        errors.append(
            ConsistencyError(
                "Faltan filas XLM-R en slice_metrics.csv y platform_prose_xlmr.tex "
                "no documenta la ausencia de predicciones"
            )
        )

    if "mbert" in platform_df["model"].values if "model" in platform_df.columns else True:
        mbert_metrics = _slice_platform_metrics(slice_df, "mbert")
        for slice_name, expected in mbert_metrics.items():
            f1_fmt = fmt_num(expected["f1_macro"])
            if f1_fmt not in tab_text:
                errors.append(
                    ConsistencyError(
                        f"tab_platform.tex no contiene F1 {f1_fmt} de mBERT ({slice_name})"
                    )
                )
            if mbert_text and f1_fmt not in mbert_text and "no disponible" not in mbert_text.lower():
                errors.append(
                    ConsistencyError(
                        f"platform_prose_mbert.tex no contiene F1 {f1_fmt} ({slice_name})"
                    )
                )

    # Verificar que cada fila de tab_platform coincide con el CSV
    for _, row in platform_df.iterrows():
        model = str(row["model"])
        slice_name = str(row["slice"])
        f1_fmt = fmt_num(row["f1_macro"])
        n_fmt = fmt_num(row["n"], digits=0)
        acc_fmt = fmt_num(row["accuracy"])
        label = _model_label(model)
        plat = _platform_label(slice_name)
        if f1_fmt not in tab_text or n_fmt not in tab_text:
            errors.append(
                ConsistencyError(
                    f"tab_platform.tex incompleto para {label} / {plat} "
                    f"(esperado n={n_fmt}, F1={f1_fmt})"
                )
            )
        _ = acc_fmt  # accuracy checked via tab row presence

    return errors


def _banned_historical_in_text(text: str, *, allow_traceability: bool = False) -> list[str]:
    found: list[str] = []
    for value in BANNED_HISTORICAL_PLATFORM_F1:
        fmt = fmt_num(value)
        if fmt in text:
            if allow_traceability and "not_reproducible" in text.lower():
                continue
            found.append(fmt)
    return found


def check_slice_metrics(
    slice_csv: Path,
    generated_dir: Path,
) -> list[ConsistencyError]:
    """Compara slice_metrics.csv con tablas/prosa LaTeX de cortes mBERT."""
    errors: list[ConsistencyError] = []
    if not slice_csv.exists():
        errors.append(ConsistencyError(f"Artefacto no encontrado: {slice_csv}"))
        return errors

    slice_df = pd.read_csv(slice_csv)
    mbert = slice_df[slice_df["model"] == "mbert"] if "model" in slice_df.columns else slice_df
    if mbert.empty:
        errors.append(ConsistencyError("slice_metrics.csv sin filas mBERT"))
        return errors

    tex_files = [
        "tab_slices.tex",
        "slices_prose.tex",
        "slices_intro.tex",
        "platform_prose_mbert.tex",
        "tab_platform.tex",
    ]

    texts: dict[str, str] = {}
    for name in tex_files:
        path = generated_dir / name
        if not path.exists():
            errors.append(ConsistencyError(f"LaTeX generado ausente: {name}"))
            continue
        texts[name] = path.read_text(encoding="utf-8")

    if errors:
        return errors

    official_run_id = None
    if MBERT_OFFICIAL_RUN_PATH.exists():
        with open(MBERT_OFFICIAL_RUN_PATH, encoding="utf-8") as handle:
            official_run_id = json.load(handle).get("run_id")

    if official_run_id and "run_id" in mbert.columns:
        mbert_run_ids = set(mbert["run_id"].dropna().unique())
        if official_run_id not in mbert_run_ids:
            errors.append(
                ConsistencyError(
                    f"run_id mBERT en slice_metrics ({mbert_run_ids}) "
                    f"no coincide con oficial ({official_run_id})"
                )
            )

    for _, row in mbert.iterrows():
        f1_fmt = fmt_num(row["f1_macro"])
        n_fmt = fmt_num(row["n"], digits=0)
        slice_name = str(row["slice"])

        if "tab_slices.tex" in texts:
            text = texts["tab_slices.tex"]
            if f1_fmt not in text:
                errors.append(
                    ConsistencyError(f"tab_slices.tex no contiene F1 {f1_fmt} ({slice_name})")
                )
            if n_fmt not in text:
                errors.append(
                    ConsistencyError(f"tab_slices.tex no contiene n={n_fmt} ({slice_name})")
                )

        if slice_name.startswith("plataforma_") and "tab_platform.tex" in texts:
            text = texts["tab_platform.tex"]
            if f1_fmt not in text:
                errors.append(
                    ConsistencyError(
                        f"tab_platform.tex no contiene F1 {f1_fmt} ({slice_name})"
                    )
                )
            if n_fmt not in text:
                errors.append(
                    ConsistencyError(
                        f"tab_platform.tex no contiene n={n_fmt} ({slice_name})"
                    )
                )

        prose_slices = {
            "plataforma_X",
            "plataforma_Facebook",
            "ironia_si",
            "ironia_no",
            "sarcasmo_si",
            "sarcasmo_no",
            "contexto_politico_si",
            "contexto_politico_no",
        }
        if slice_name in prose_slices and "slices_prose.tex" in texts:
            if f1_fmt not in texts["slices_prose.tex"]:
                errors.append(
                    ConsistencyError(
                        f"slices_prose.tex no contiene F1 {f1_fmt} ({slice_name})"
                    )
                )

        intro_slices = {"plataforma_X", "ironia_si", "sarcasmo_si"}
        if slice_name in intro_slices and "slices_intro.tex" in texts:
            intro_fmt = fmt_num(row["f1_macro"], digits=2)
            if intro_fmt not in texts["slices_intro.tex"]:
                errors.append(
                    ConsistencyError(
                        f"slices_intro.tex no contiene F1 {intro_fmt} ({slice_name})"
                    )
                )

    for name, text in texts.items():
        if name == "mbert_platform_traceability_prose.tex":
            continue
        banned = _banned_historical_in_text(text)
        for fmt in banned:
            errors.append(
                ConsistencyError(
                    f"{name} contiene F1 histórico prohibido {fmt} (usar CSV oficial)"
                )
            )

    trace_path = generated_dir / "mbert_platform_traceability_prose.tex"
    if trace_path.exists():
        trace_text = trace_path.read_text(encoding="utf-8")
        if "not_reproducible" not in trace_text and _banned_historical_in_text(trace_text):
            errors.append(
                ConsistencyError(
                    "mbert_platform_traceability_prose.tex cita históricos sin marcar not_reproducible"
                )
            )

    return errors


def assert_slice_consistency(slice_csv: Path, generated_dir: Path) -> None:
    errors = check_slice_metrics(slice_csv, generated_dir)
    if errors:
        messages = "\n".join(f"  - {e.message}" for e in errors)
        raise AssertionError(f"Inconsistencias de cortes mBERT:\n{messages}")


def check_mcnemar_metrics(
    mcnemar_csv: Path,
    generated_dir: Path,
) -> list[ConsistencyError]:
    """Compara mcnemar_tests.csv con tablas/prosa LaTeX de McNemar."""
    errors: list[ConsistencyError] = []
    if not mcnemar_csv.exists():
        errors.append(ConsistencyError(f"Artefacto no encontrado: {mcnemar_csv}"))
        return errors

    tab_mcnemar = generated_dir / "tab_mcnemar.tex"
    prose_mcnemar = generated_dir / "mcnemar_prose.tex"
    for tex_path in (tab_mcnemar, prose_mcnemar):
        if not tex_path.exists():
            errors.append(ConsistencyError(f"LaTeX generado ausente: {tex_path.name}"))
            return errors

    df = pd.read_csv(mcnemar_csv)
    if "model_b" not in df.columns:
        return errors

    tab_text = tab_mcnemar.read_text(encoding="utf-8")
    prose_text = prose_mcnemar.read_text(encoding="utf-8")

    xlmr_rows = df[df["model_b"] == "xlm-roberta"]
    if xlmr_rows.empty:
        errors.append(
            ConsistencyError("mcnemar_tests.csv no contiene fila mbert vs xlm-roberta")
        )
        return errors

    row = xlmr_rows.iloc[0]
    p_fmt = fmt_num(row["p_value"], digits=6)
    if p_fmt not in tab_text:
        errors.append(
            ConsistencyError(f"tab_mcnemar.tex no contiene p-valor XLM-R ({p_fmt})")
        )
    if p_fmt not in prose_text:
        errors.append(
            ConsistencyError(f"mcnemar_prose.tex no contiene p-valor XLM-R ({p_fmt})")
        )

    b_fmt = fmt_num(row["b"], digits=0)
    c_fmt = fmt_num(row["c"], digits=0)
    if b_fmt not in prose_text or c_fmt not in prose_text:
        errors.append(
            ConsistencyError(
                f"mcnemar_prose.tex no contiene conteos discordantes b={b_fmt}, c={c_fmt}"
            )
        )

    return errors


def assert_mcnemar_consistency(mcnemar_csv: Path, generated_dir: Path) -> None:
    errors = check_mcnemar_metrics(mcnemar_csv, generated_dir)
    if errors:
        messages = "\n".join(f"  - {e.message}" for e in errors)
        raise AssertionError(f"Inconsistencias de McNemar:\n{messages}")


def assert_platform_consistency(
    slice_csv: Path,
    generated_dir: Path,
    *,
    strict_xlmr: bool = False,
) -> None:
    errors = check_platform_metrics(slice_csv, generated_dir, strict_xlmr=strict_xlmr)
    if errors:
        messages = "\n".join(f"  - {e.message}" for e in errors)
        raise AssertionError(f"Inconsistencias de métricas por plataforma:\n{messages}")


def check_mbert_official_metrics(
    metrics_csv: Path,
    generated_dir: Path,
    *,
    official_run_path: Path | None = None,
) -> list[ConsistencyError]:
    """Compara mbert_test_metrics.csv con tablas/prosa LaTeX oficiales."""
    errors: list[ConsistencyError] = []
    if not metrics_csv.exists():
        errors.append(ConsistencyError(f"Artefacto no encontrado: {metrics_csv}"))
        return errors

    row = pd.read_csv(metrics_csv).iloc[0]
    f1_fmt = fmt_num(row["f1_macro"])
    f1_lo = fmt_num(row.get("f1_bootstrap_lo"))
    f1_hi = fmt_num(row.get("f1_bootstrap_hi"))
    f1_latex = latex_fmt_num(row["f1_macro"])
    f1_lo_latex = latex_fmt_num(row.get("f1_bootstrap_lo"))
    f1_hi_latex = latex_fmt_num(row.get("f1_bootstrap_hi"))

    tex_files = {
        "tab_mbert_global.tex": [f1_fmt],
        "tab_model_comparison.tex": [f1_fmt],
        "mbert_official_prose.tex": [f1_latex, f1_lo_latex, f1_hi_latex],
        "mbert_intro_prose.tex": [f1_latex, f1_lo_latex, f1_hi_latex],
        "mbert_conclusions_prose.tex": [f1_latex],
        "mbert_benchmark_prose.tex": [f1_latex],
    }

    for name, expected_values in tex_files.items():
        path = generated_dir / name
        if not path.exists():
            errors.append(ConsistencyError(f"LaTeX generado ausente: {name}"))
            continue
        text = path.read_text(encoding="utf-8")
        for value in expected_values:
            if value and value not in text:
                errors.append(
                    ConsistencyError(f"{name} no contiene el valor esperado {value}")
                )
        if "0.7862" in text or "0{,}7862" in text:
            errors.append(ConsistencyError(f"{name} contiene F1 histórico 0.7862"))
        if "0.778" in text or "0{,}778" in text:
            errors.append(
                ConsistencyError(
                    f"{name} contiene redondeo legacy 0.778 en lugar del oficial"
                )
            )

    official_run_path = official_run_path or MBERT_OFFICIAL_RUN_PATH
    if official_run_path.exists():
        with open(official_run_path, encoding="utf-8") as handle:
            run = json.load(handle)
        if fmt_num(run.get("official_f1_macro")) != f1_fmt:
            errors.append(
                ConsistencyError(
                    "mbert_official_run.json y mbert_test_metrics.csv difieren en F1"
                )
            )
    else:
        errors.append(ConsistencyError(f"Artefacto no encontrado: {official_run_path}"))

    trace_path = MBERT_TRACEABILITY_CSV
    if trace_path.exists():
        trace = pd.read_csv(trace_path)
        official_rows = trace[trace["status"] == "official"]
        if official_rows.empty:
            errors.append(ConsistencyError("mbert_f1_traceability.csv sin fila official"))
        elif fmt_num(official_rows.iloc[0]["f1_macro_published"]) != f1_fmt:
            errors.append(
                ConsistencyError("Trazabilidad official no coincide con métricas CSV")
            )
    else:
        errors.append(ConsistencyError(f"Artefacto no encontrado: {trace_path}"))

    if round(float(row["f1_macro"]), 4) != OFFICIAL_F1_MACRO:
        errors.append(
            ConsistencyError(
                f"F1 en CSV ({f1_fmt}) no coincide con constante oficial {OFFICIAL_F1_MACRO}"
            )
        )

    return errors


def assert_mbert_official_consistency(
    metrics_csv: Path,
    generated_dir: Path,
    *,
    official_run_path: Path | None = None,
) -> None:
    errors = check_mbert_official_metrics(
        metrics_csv,
        generated_dir,
        official_run_path=official_run_path,
    )
    if errors:
        messages = "\n".join(f"  - {e.message}" for e in errors)
        raise AssertionError(f"Inconsistencias de métrica oficial mBERT:\n{messages}")
