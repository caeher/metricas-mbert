"""Benchmark zero-shot contra datasets públicos de Hugging Face."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

from discurso_odio.config import get_hf_token, load_settings
from discurso_odio.models.mbert import predict

logger = logging.getLogger(__name__)

# Mapeo de etiquetas externas -> label ordinal del proyecto (0-3)
# None = excluir fila


def _map_piuba(row: dict) -> int | None:
    hateful = row.get("HATEFUL")
    if hateful is None:
        hateful = row.get("hateful")
    calls = row.get("CALLS")
    if calls is None:
        calls = row.get("calls")
    if hateful is None:
        return None
    hateful = int(hateful)
    calls = int(calls) if calls is not None else 0
    if hateful == 0:
        return 0
    return 3 if calls == 1 else 2


def _map_socialtox(row: dict) -> int | None:
    tox = row.get("TOXICIDAD") or row.get("toxicity") or row.get("Toxicity") or row.get("label")
    if tox is None:
        return None
    tox = str(tox).lower()
    if "no t" in tox or tox.startswith("1"):
        return 0
    if "liger" in tox or tox.startswith("2"):
        return 1
    return 2


def _map_sinodio(row: dict) -> int | None:
    label = row.get("label")
    if label is None:
        label = row.get("hate_speech") or row.get("class")
    if label is None:
        return None
    if isinstance(label, (int, float)):
        return 0 if int(label) == 0 else 2
    label = str(label).lower()
    if label in ("0", "no_hate", "no", "non-hate", "non_hate"):
        return 0
    return 2


def _map_textdetox(row: dict) -> int | None:
    toxic = row.get("toxic")
    if toxic is None:
        toxic = row.get("label")
    if toxic is None:
        return None
    return 1 if int(toxic) == 1 else 0


def _map_alia(row: dict) -> int | None:
    label = row.get("foe_class")
    if label is None:
        label = row.get("label")
    if label is None:
        return None
    return 2 if int(label) == 1 else 0


def _map_exist(row: dict) -> int | None:
    label = row.get("label")
    if label is None:
        label = row.get("sexist")
    if label is None:
        return None
    if isinstance(label, (int, float)):
        return 0 if int(label) == 0 else 2
    label = str(label).lower()
    if label in ("0", "non-sexist", "non_sexist", "no"):
        return 0
    return 2


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "piuba-bigdata/contextualized_hate_speech": {
        "text_col": "text",
        "mapper": _map_piuba,
        "split": "test",
    },
    "gplsi/SocialTOX": {
        "text_col": "text",
        "mapper": _map_socialtox,
        "split": "test",
    },
    "antonn-dromundo/SinOdio-LATAM-Regional-HateSpeech": {
        "text_col": "text",
        "mapper": _map_sinodio,
        "split": "test",
    },
    "textdetox/multilingual_toxicity_dataset": {
        "text_col": "text",
        "mapper": _map_textdetox,
        "split": "es",
    },
    "SINAI/ALIA-es-discriminative-hate-speech": {
        "text_col": "text",
        "mapper": _map_alia,
        "config": "human_sample",
        "split": "test",
    },
    "Gerard-1705/exist_dataset_es": {
        "text_col": "text",
        "mapper": _map_exist,
        "split": "test",
    },
}


def _get_text_col(ds, registry: dict) -> str:
    col = registry.get("text_col", "text")
    if col in ds.column_names:
        return col
    for candidate in ("text", "tweet", "comment", "content", "Text"):
        if candidate in ds.column_names:
            return candidate
    raise ValueError(f"No se encontró columna de texto en {ds.column_names}")


def _stratified_subsample(df: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_samples:
        return df
    n_classes = df["label"].nunique()
    per_class = max(1, max_samples // n_classes)
    parts = []
    for _, group in df.groupby("label"):
        parts.append(group.sample(n=min(len(group), per_class), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_samples:
        out = out.sample(n=max_samples, random_state=seed)
    return out


def load_external_dataset(dataset_id: str, cfg: dict | None = None) -> pd.DataFrame:
    registry = DATASET_REGISTRY.get(dataset_id, {})
    token = get_hf_token()
    config_name = cfg.get("config") if cfg else registry.get("config")
    split = (cfg or registry).get("split", "test")

    load_kwargs: dict[str, Any] = {"path": dataset_id, "split": split}
    if config_name:
        load_kwargs["name"] = config_name
    if token:
        load_kwargs["token"] = token

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        logger.warning("No se pudo cargar %s: %s", dataset_id, e)
        return pd.DataFrame(columns=["text", "label"])

    text_col = _get_text_col(ds, registry)
    mapper = registry.get("mapper")

    rows = []
    for item in ds:
        label = mapper(item) if mapper else None
        if label is None:
            continue
        text = item.get(text_col, "")
        if not str(text).strip():
            continue
        rows.append({"text": str(text), "label": int(label)})

    if not rows:
        return pd.DataFrame(columns=["text", "label"])

    df = pd.DataFrame(rows)
    max_samples = (cfg or {}).get("max_samples")
    if max_samples and len(df) > max_samples:
        df = _stratified_subsample(df, max_samples)

    return df


def project_prediction(pred: int) -> dict[str, int]:
    """Proyecta predicción 4-clases al esquema binario unificado."""
    return {
        "binary_toxic": 1 if pred >= 1 else 0,
        "ordinal": pred,
    }


def run_benchmark(
    model_dir: str,
    dataset_configs: list[dict] | None = None,
) -> pd.DataFrame:
    settings = load_settings()
    configs = dataset_configs or settings["benchmark"]["default_datasets"]
    results = []

    for cfg in configs:
        dataset_id = cfg["id"]
        logger.info("Benchmarking %s...", dataset_id)
        df = load_external_dataset(dataset_id, cfg)
        if df.empty:
            results.append({
                "dataset": dataset_id,
                "n_samples": 0,
                "status": "skipped",
                "f1_macro": None,
                "f1_binary": None,
                "accuracy": None,
            })
            continue

        preds, probs = predict(df["text"].tolist(), model_dir)
        y_true = df["label"].values
        y_pred = preds

        binary_true = (y_true >= 1).astype(int)
        binary_pred = (y_pred >= 1).astype(int)

        results.append({
            "dataset": dataset_id,
            "n_samples": len(df),
            "status": "ok",
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_binary": f1_score(binary_true, binary_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "accuracy_binary": accuracy_score(binary_true, binary_pred),
        })

    return pd.DataFrame(results)
