"""Normalización centralizada de indicadores SI/SÍ/NO para cortes de evaluación."""

from __future__ import annotations

import unicodedata

import pandas as pd

POSITIVE_VALUES = frozenset({"SI", "YES", "TRUE", "1"})
NEGATIVE_VALUES = frozenset({"NO", "FALSE", "0", "N"})

SLICE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Presencia_Sarcasmo": ("Presencia_Sarcasmo", "sarcasmo"),
    "Presencia_Ironia": ("Presencia_Ironia", "ironia"),
    "Contexto_Politico": ("Contexto_Politico", "contexto_politico"),
}


def _fold_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _canonicalize(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = _fold_accents(str(value).strip().upper())
    return text


def normalize_indicator(series: pd.Series) -> pd.Series:
    """Clasifica indicadores en True (positivo), False (negativo) o NA (excluido).

    Positivos: SI, SÍ, YES, TRUE, 1.
    Negativos: NO, FALSE, 0, N.
    Cualquier otro valor queda como NA y no entra en cortes sí/no.
    """
    canonical = series.map(_canonicalize)
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[canonical.isin(POSITIVE_VALUES)] = True
    result[canonical.isin(NEGATIVE_VALUES)] = False
    return result


def resolve_slice_columns(df: pd.DataFrame) -> dict[str, str]:
    """Resuelve nombres canónicos de columnas de cortes presentes en *df*."""
    resolved: dict[str, str] = {}
    for canonical, aliases in SLICE_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                resolved[canonical] = alias
                break
    return resolved
