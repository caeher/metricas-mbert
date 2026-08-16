"""Contrato único de preprocesamiento e inferencia (max_length + normalizador)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from discurso_odio.config import load_settings
from discurso_odio.features.normalization import normalize_for_model

CONTRACT_FILENAME = "inference_contract.json"

NORMALIZER_REGISTRY: dict[str, Callable[[str], str]] = {
    "normalize_for_model": normalize_for_model,
}


@dataclass(frozen=True)
class InferenceContract:
    max_length: int
    text_column: str
    normalizer: str
    normalizer_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "text_column": self.text_column,
            "normalizer": self.normalizer,
            "normalizer_version": self.normalizer_version,
        }

    def metadata(self) -> dict[str, Any]:
        """Metadatos para artefactos de métricas y predicciones."""
        return {
            "max_length": self.max_length,
            "text_column": self.text_column,
            "normalizer": self.normalizer,
            "normalizer_version": self.normalizer_version,
        }

    def get_normalizer_fn(self) -> Callable[[str], str]:
        if self.normalizer not in NORMALIZER_REGISTRY:
            raise ValueError(
                f"Normalizador desconocido: {self.normalizer!r}. "
                f"Disponibles: {list(NORMALIZER_REGISTRY)}"
            )
        return NORMALIZER_REGISTRY[self.normalizer]

    def validate_max_length(self, max_length: int | None) -> int:
        """Valida que un override opcional coincida con el contrato."""
        if max_length is None:
            return self.max_length
        if max_length != self.max_length:
            raise ValueError(
                f"max_length={max_length} no coincide con el contrato ({self.max_length})"
            )
        return self.max_length


def _contract_from_dict(data: dict[str, Any]) -> InferenceContract:
    required = ("max_length", "text_column", "normalizer", "normalizer_version")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Contrato incompleto, faltan campos: {missing}")
    return InferenceContract(
        max_length=int(data["max_length"]),
        text_column=str(data["text_column"]),
        normalizer=str(data["normalizer"]),
        normalizer_version=str(data["normalizer_version"]),
    )


def get_inference_contract() -> InferenceContract:
    """Lee el contrato canónico desde config/settings.yaml."""
    settings = load_settings()
    if "inference_contract" not in settings:
        raise KeyError(
            "Falta sección 'inference_contract' en config/settings.yaml"
        )
    return _contract_from_dict(settings["inference_contract"])


def apply_normalization(texts: list[str], contract: InferenceContract | None = None) -> list[str]:
    """Aplica el normalizador del contrato a una lista de textos."""
    contract = contract or get_inference_contract()
    fn = contract.get_normalizer_fn()
    return [fn(t) if isinstance(t, str) else fn("") for t in texts]


def apply_normalization_single(text: str, contract: InferenceContract | None = None) -> str:
    """Aplica el normalizador del contrato a un solo texto."""
    return apply_normalization([text], contract)[0]


def write_inference_contract(model_dir: Path | str, contract: InferenceContract | None = None) -> Path:
    """Persiste inference_contract.json junto al checkpoint."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    contract = contract or get_inference_contract()
    path = model_dir / CONTRACT_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contract.as_dict(), f, indent=2, ensure_ascii=False)
    return path


def load_inference_contract(model_dir: Path | str) -> InferenceContract | None:
    """Carga el contrato desde un checkpoint si existe."""
    path = Path(model_dir) / CONTRACT_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return _contract_from_dict(json.load(f))


def resolve_inference_contract(model_dir: Path | str | None = None) -> InferenceContract:
    """Resuelve contrato: checkpoint > settings.yaml."""
    if model_dir is not None:
        loaded = load_inference_contract(model_dir)
        if loaded is not None:
            return loaded
    return get_inference_contract()
