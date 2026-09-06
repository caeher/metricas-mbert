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
    settings = load_settings()
    names = settings["labels"]["names"]
    pesos = metadata["pesos_clase"]
    return {i: float(pesos[name]) for i, name in enumerate(names)}


def compute_sample_loss_weights(
    train_df: pd.DataFrame,
    *,
    threat_synthetic_effective_n: float = 90.0,
    origen_col: str = "origen",
) -> pd.Series:
    """Peso por fila: amenazas sintéticas con masa efectiva total ~threat_synthetic_effective_n."""
    if origen_col not in train_df.columns:
        return pd.Series(1.0, index=train_df.index)

    origins = train_df[origen_col].astype(str)
    threat_mask = origins.isin(THREAT_SYNTHETIC_ORIGINS)
    n_threat = int(threat_mask.sum())
    if n_threat == 0:
        return pd.Series(1.0, index=train_df.index)

    threat_weight = float(threat_synthetic_effective_n) / n_threat
    weights = pd.Series(1.0, index=train_df.index, dtype=float)
    weights.loc[threat_mask] = threat_weight
    return weights


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
