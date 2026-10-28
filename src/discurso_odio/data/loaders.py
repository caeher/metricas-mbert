"""Carga de datasets locales."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from discurso_odio.config import load_settings
from discurso_odio.paths import (
    CORPUS_CSV,
    DATASET_METADATA_JSON,
    TEST_ADVERSARIAL_CSV,
    TEST_ADVERSARIAL_144_CSV,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)

THREAT_SYNTHETIC_ORIGINS = frozenset(
    {"sintetico_amenaza_ofuscada", "sintetico_emoji_amenaza"}
)

# Familias sintéticas más "de plantilla" (ablación Fase 1)
TEMPLATE_HEAVY_ORIGINS = frozenset(
    {
        "sintetico_relleno_emoji",
        "sintetico_emoji_odio",
        "sintetico_emoji_ofensivo",
        "sintetico_emoji_notoxico",
        "sintetico_emoji_amenaza",
    }
)

REAL_ORIGIN = "real"


def load_split(name: str) -> pd.DataFrame:
    paths = {
        "train": TRAIN_CSV,
        "val": VAL_CSV,
        "test": TEST_CSV,
        "corpus": CORPUS_CSV,
        "adversarial": TEST_ADVERSARIAL_CSV,
        "adversarial_144": TEST_ADVERSARIAL_144_CSV,
    }
    if name not in paths:
        raise ValueError(f"Split desconocido: {name}. Use {list(paths)}")
    path = paths[name]
    if not path.exists():
        raise FileNotFoundError(f"Split no encontrado: {path}")
    return pd.read_csv(path)


def load_all_splits() -> dict[str, pd.DataFrame]:
    return {split: load_split(split) for split in ("train", "val", "test", "corpus")}


@lru_cache(maxsize=1)
def load_b2_metadata() -> dict[str, Any]:
    with open(DATASET_METADATA_JSON, encoding="utf-8") as handle:
        return json.load(handle)


def get_b2_class_weights() -> dict[int, float]:
    """Pesos de clase fijos del dataset B2 (metadata.json), no recalculados."""
    metadata = load_b2_metadata()
    version = str(metadata.get("version", "")).casefold()
    if version not in {"b2", "1.1.0"} and "pesos_clase" not in metadata:
        raise ValueError(
            "metadata.json no trae pesos B2 (version "
            f"{metadata.get('version')!r}). Use get_class_weights(train_df) para v1.8+."
        )
    settings = load_settings()
    names = settings["labels"]["names"]
    pesos = metadata["pesos_clase"]
    return {i: float(pesos[name]) for i, name in enumerate(names)}


def filter_train_by_synthetic_policy(
    train_df: pd.DataFrame,
    policy: str = "b2_full",
    *,
    origen_col: str = "origen",
) -> pd.DataFrame:
    """Filtra train según política de sintéticos (sin datos nuevos)."""
    if origen_col not in train_df.columns:
        return train_df.copy()

    origins = train_df[origen_col].astype(str)
    if policy == "b2_full":
        return train_df.copy()
    if policy == "real_only":
        return train_df.loc[origins == REAL_ORIGIN].reset_index(drop=True)
    if policy == "b2_no_template_heavy":
        mask = (origins == REAL_ORIGIN) | (~origins.isin(TEMPLATE_HEAVY_ORIGINS))
        return train_df.loc[mask].reset_index(drop=True)
    raise ValueError(
        f"Política desconocida: {policy}. Use b2_full, real_only, b2_no_template_heavy."
    )


def compute_sample_loss_weights(
    train_df: pd.DataFrame,
    *,
    threat_synthetic_effective_n: float = 90.0,
    synthetic_effective_fraction: float | None = None,
    platform_class_boost: dict[tuple[str, int], float] | None = None,
    origen_col: str = "origen",
    platform_col: str = "Plataforma",
    label_col: str = "label",
) -> pd.Series:
    """Peso por fila: amenazas sintéticas, downweight global y boost plataforma×clase."""
    weights = pd.Series(1.0, index=train_df.index, dtype=float)

    if origen_col in train_df.columns:
        origins = train_df[origen_col].astype(str)
        synth_mask = origins != REAL_ORIGIN
        threat_mask = origins.isin(THREAT_SYNTHETIC_ORIGINS)

        if synthetic_effective_fraction is not None and synth_mask.any():
            weights.loc[synth_mask] = float(synthetic_effective_fraction)

        n_threat = int(threat_mask.sum())
        if n_threat > 0:
            threat_weight = float(threat_synthetic_effective_n) / n_threat
            weights.loc[threat_mask] = threat_weight

    if platform_class_boost:
        for (platform, label), factor in platform_class_boost.items():
            if platform_col not in train_df.columns:
                break
            mask = (
                (train_df[platform_col].astype(str) == platform)
                & (train_df[label_col].astype(int) == int(label))
            )
            if origen_col in train_df.columns:
                mask &= train_df[origen_col].astype(str) == REAL_ORIGIN
            weights.loc[mask] *= float(factor)

    return weights


def default_facebook_boost_weights() -> dict[tuple[str, int], float]:
    """Boost recomendado Fase 1: subir odio/amenazas FB, bajar ofensivo FB."""
    return {
        ("Facebook", 2): 2.5,
        ("Facebook", 3): 1.8,
        ("Facebook", 1): 0.85,
    }


def summarize_train_platform_class_weights(
    train_df: pd.DataFrame | None = None,
    *,
    origen_col: str = "origen",
    platform_col: str = "Plataforma",
    label_col: str = "label",
) -> dict[str, Any]:
    """Resumen real vs sintético por plataforma×clase para error analysis."""
    train_df = train_df if train_df is not None else load_split("train")
    rows: list[dict[str, Any]] = []
    for (platform, label), group in train_df.groupby([platform_col, label_col]):
        origins = group[origen_col].astype(str) if origen_col in group.columns else pd.Series(REAL_ORIGIN)
        n_real = int((origins == REAL_ORIGIN).sum())
        n_synth = int((origins != REAL_ORIGIN).sum())
        rows.append(
            {
                "platform": str(platform),
                "label": int(label),
                "n_total": int(len(group)),
                "n_real": n_real,
                "n_synthetic": n_synth,
                "synthetic_ratio": float(n_synth / len(group)) if len(group) else 0.0,
            }
        )
    return {"rows": rows, "n_train": int(len(train_df))}


def load_train_excluding_adversarial_overlap() -> pd.DataFrame:
    """Train B2 sin filas cuyo texto coincide con la batería adversarial (T-06)."""
    from discurso_odio.evaluation.adversarial import adversarial_text_set, find_train_overlaps

    train_df = load_split("train")
    overlaps = find_train_overlaps(train_df)
    if overlaps.empty:
        return train_df
    adv_texts = adversarial_text_set()
    mask = ~train_df["texto_modelo"].astype(str).str.strip().str.lower().isin(adv_texts)
    return train_df.loc[mask].reset_index(drop=True)


def get_class_weights(train_df: pd.DataFrame, label_col: str = "label") -> dict[int, float]:
    """Pesos inversamente proporcionales a la frecuencia (solo sobre train)."""
    counts = train_df[label_col].value_counts().sort_index()
    n = len(train_df)
    k = len(counts)
    return {int(cls): n / (k * cnt) for cls, cnt in counts.items()}
