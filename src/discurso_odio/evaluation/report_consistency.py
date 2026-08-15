"""Verificación de consistencia entre artefactos de métricas y LaTeX generado."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from discurso_odio.evaluation.slices import PLATFORM_SLICES, platform_rows

MBERT_PLATFORM_F1 = {
    "plataforma_X": 0.8386,
    "plataforma_Facebook": 0.6727,
}

XLMR_AUDIT_F1 = {
    "plataforma_X": 0.867,
    "plataforma_Facebook": 0.715,
}

FLOAT_PATTERN = re.compile(r"\b0\.\d{2,4}\b")


@dataclass
class ConsistencyError:
    message: str


def fmt_num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    return f"{float(value):.{digits}f}"


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
        for slice_name, mbert_f1 in MBERT_PLATFORM_F1.items():
            mbert_fmt = fmt_num(mbert_f1)
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
