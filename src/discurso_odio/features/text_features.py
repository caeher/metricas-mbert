"""Feature engineering numérico y binario."""

from __future__ import annotations

import pandas as pd

from discurso_odio.features.normalization import (
    has_accents_or_enye,
    has_mention_literal,
    has_punctuation,
    has_uppercase,
    has_url_literal,
)


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["SI", "SÍ", "YES", "TRUE", "1"])


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera features a partir del dataframe; reutiliza columnas existentes si están."""
    out = df.copy()

    if "n_palabras" not in out.columns and "texto_modelo" in out.columns:
        out["n_palabras"] = out["texto_modelo"].str.split().str.len()
    if "n_chars" not in out.columns and "texto_modelo" in out.columns:
        out["n_chars"] = out["texto_modelo"].str.len()

    if "texto_modelo" in out.columns:
        out["upper_ratio"] = out["texto_modelo"].apply(
            lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
        )
        out["exclamation_count"] = out["texto_modelo"].str.count("!")
        out["question_count"] = out["texto_modelo"].str.count(r"\?")
        out["has_url"] = out["texto_modelo"].apply(has_url_literal).astype(int)
        out["has_mention"] = out["texto_modelo"].apply(has_mention_literal).astype(int)
        out["has_accents"] = out["texto_modelo"].apply(has_accents_or_enye).astype(int)
        out["has_upper"] = out["texto_modelo"].apply(has_uppercase).astype(int)
        out["has_punct"] = out["texto_modelo"].apply(has_punctuation).astype(int)

    if "Plataforma" in out.columns:
        out["es_x"] = (out["Plataforma"].str.upper() == "X").astype(int)

    for col, out_col in [
        ("Presencia_Sarcasmo", "sarcasmo"),
        ("Presencia_Ironia", "ironia"),
        ("Contexto_Politico", "contexto_politico"),
    ]:
        if col in out.columns:
            out[out_col] = _to_bool(out[col]).astype(int)

    if "n_jerga" in out.columns:
        out["dialecto_salvadoreno"] = (out["n_jerga"] > 0).astype(int)

    return out
