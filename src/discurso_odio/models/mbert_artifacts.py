"""Constantes y helpers para el paquete de auditoría mBERT (T-02)."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from discurso_odio.config import load_settings
from discurso_odio.evaluation.predictions import git_commit
from discurso_odio.models.inference_contract import get_inference_contract, write_inference_contract
from discurso_odio.models.mbert import default_mbert_model_dir
from discurso_odio.paths import PROJECT_ROOT, TEST_CSV

PACKAGE_ROOT_NAME = "mbert-sv"
ZIP_NAME = "mbert-sv-audit.zip"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MANIFEST_REPORT_PATH = PROJECT_ROOT / "reports" / "metrics" / "mbert_checkpoint_manifest.json"
VERIFY_REPORT_PATH = PROJECT_ROOT / "reports" / "metrics" / "mbert_checkpoint_verify.json"

F1_TOLERANCE = 1e-4
OFFICIAL_F1_MACRO = 0.7777
BASE_MODEL = "bert-base-multilingual-cased"

REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
OPTIONAL_FILES = (
    "training_args.bin",
    "special_tokens_map.json",
    "vocab.txt",
    "train_metrics.json",
    "test_metrics.json",
)
GENERATED_FILES = (
    "inference_contract.json",
    "MANIFEST.json",
    "README.md",
)
EXCLUDED_FILES = {
    "optimizer.pt",
    "optimizer-002.pt",
    "scheduler.pt",
    "rng_state.pth",
}
MANIFEST_REQUIRED_FIELDS = (
    "git_commit",
    "packaged_at",
    "seed",
    "base_model",
    "checkpoint",
    "epoch",
    "global_step",
    "sha256",
    "max_length",
    "text_column",
    "normalizer",
    "normalizer_version",
    "test_csv",
    "test_n",
    "test_sha256",
    "official_f1_macro",
    "f1_tolerance",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mbert_source(explicit: Path | None = None) -> Path:
    if explicit is not None:
        source = Path(explicit)
        if not (source / "model.safetensors").exists():
            raise FileNotFoundError(f"No hay model.safetensors en {source}")
        return source

    candidates = [
        default_mbert_model_dir(),
        PROJECT_ROOT / "resultados-metricas" / "models" / "mbert-sv",
    ]
    for candidate in candidates:
        if (candidate / "model.safetensors").exists() and (candidate / "config.json").exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró un checkpoint mBERT. Se esperaba models/mbert-sv/ "
        "o resultados-metricas/models/mbert-sv/."
    )


def list_package_files(source_dir: Path) -> list[str]:
    names: list[str] = []
    for name in REQUIRED_FILES:
        if not (source_dir / name).exists():
            raise FileNotFoundError(f"Falta archivo requerido en el checkpoint: {name}")
        names.append(name)
    for name in OPTIONAL_FILES:
        if (source_dir / name).exists():
            names.append(name)
    return names


def apply_id2label(config_path: Path, label_names: list[str] | None = None) -> dict[str, Any]:
    settings = load_settings()
    names = label_names or list(settings["labels"]["names"])
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["id2label"] = {str(i): name for i, name in enumerate(names)}
    config["label2id"] = {name: i for i, name in enumerate(names)}
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return config


def read_checkpoint_identity(source_dir: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "checkpoint": "mbert-sv",
        "epoch": None,
        "global_step": None,
        "best_metric_val_f1_macro": None,
        "best_model_checkpoint": None,
    }
    states = sorted(source_dir.glob("checkpoint-*/trainer_state.json"))
    chosen: Path | None = None
    if states:
        chosen = states[-1]
        for state_path in states:
            if "checkpoint-1072" in str(state_path):
                chosen = state_path
                break
    root_state = source_dir / "trainer_state.json"
    if chosen is None and root_state.exists():
        chosen = root_state
    if chosen is None:
        return identity

    with open(chosen, encoding="utf-8") as handle:
        state = json.load(handle)
    identity["epoch"] = state.get("epoch")
    identity["global_step"] = state.get("best_global_step") or state.get("global_step")
    identity["best_metric_val_f1_macro"] = state.get("best_metric")
    best_ckpt = state.get("best_model_checkpoint")
    if best_ckpt:
        identity["best_model_checkpoint"] = Path(str(best_ckpt)).name
        identity["checkpoint"] = Path(str(best_ckpt)).name
    elif chosen.parent.name.startswith("checkpoint-"):
        identity["checkpoint"] = chosen.parent.name
    return identity


def read_official_f1(source_dir: Path) -> float:
    metrics_path = source_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        if "f1_macro" in metrics:
            return round(float(metrics["f1_macro"]), 4)
    return OFFICIAL_F1_MACRO


def package_readme() -> str:
    return """# Paquete de auditoría mBERT

Checkpoint fine-tuned de `bert-base-multilingual-cased` para clasificación
multiclase de toxicidad (El Salvador). Este ZIP es autocontenido: no requiere
archivos locales no declarados para cargar tokenizer y pesos.

## Archivos

- `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `inference_contract.json` — `max_length=128`, columna `texto_modelo`,
  normalizador `normalize_for_model` v1.1
- `MANIFEST.json` — commit, semilla, época, SHA-256, hash de `test.csv`

**Corpus B2:** `test.csv` incluye `origen` y `plantilla_id`; IDs, textos y
etiquetas son idénticos al v1. El checkpoint (`checkpoint-1072`) se entrenó
sobre train v1; el F1 oficial 0.7777 no cambia hasta reentrenar.

## Inferencia (misma config que entrenamiento/evaluación)

Desde la raíz del repositorio, con el ZIP descomprimido:

```powershell
python scripts/export_mbert_predictions.py --model-dir <ruta>/mbert-sv
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip
python scripts/inferencia_mbert.py --model-dir <ruta>/mbert-sv
```

Contrato:

- Texto: columna `texto_modelo` (o texto crudo + `normalize_for_model`)
- `max_length`: 128
- Etiquetas: 0 No Tóxico, 1 Lenguaje Ofensivo, 2 Discurso de Odio, 3 Amenazas/Violencia

Carga mínima (sin el resto del repo, salvo el normalizador si el texto no está
preprocesado):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("mbert-sv")
model = AutoModelForSequenceClassification.from_pretrained("mbert-sv")
```

Los pesos no se versionan en Git. Distribución: `artifacts/mbert-sv-audit.zip`
y el canal de artefactos (Google Drive) indicado en el README del proyecto.
"""


def build_manifest(
    *,
    staged_dir: Path,
    source_dir: Path,
    file_names: list[str],
    artifact_url: str = "",
) -> dict[str, Any]:
    settings = load_settings()
    contract = get_inference_contract()
    identity = read_checkpoint_identity(source_dir)
    hashed_names = [name for name in file_names if name != "MANIFEST.json"]
    sha_map = {name: sha256_file(staged_dir / name) for name in hashed_names}
    return {
        "git_commit": git_commit(),
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "seed": settings["project"]["seed"],
        "base_model": settings["model"]["name"],
        "checkpoint": identity["checkpoint"],
        "epoch": identity["epoch"],
        "global_step": identity["global_step"],
        "best_metric_val_f1_macro": identity["best_metric_val_f1_macro"],
        "best_model_checkpoint": identity["best_model_checkpoint"],
        "source_dir": str(source_dir.resolve()),
        "sha256": sha_map,
        "max_length": contract.max_length,
        "text_column": contract.text_column,
        "normalizer": contract.normalizer,
        "normalizer_version": contract.normalizer_version,
        "test_csv": str(TEST_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_n": 460,
        "test_sha256": sha256_file(TEST_CSV),
        "official_f1_macro": read_official_f1(source_dir),
        "f1_tolerance": F1_TOLERANCE,
        "num_labels": settings["labels"]["num_classes"],
        "labels": list(settings["labels"]["names"]),
        "artifact_channel": "Google Drive (tesina / auditoría)",
        "artifact_url": artifact_url,
        "artifact_local": f"artifacts/{ZIP_NAME}",
    }


def prepare_staging(source_dir: Path, staging_dir: Path) -> list[str]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    names = list_package_files(source_dir)
    for name in names:
        shutil.copy2(source_dir / name, staging_dir / name)
    write_inference_contract(staging_dir)
    apply_id2label(staging_dir / "config.json")
    (staging_dir / "README.md").write_text(package_readme(), encoding="utf-8")
    staged = sorted(
        path.name for path in staging_dir.iterdir() if path.is_file() and path.name != "MANIFEST.json"
    )
    return staged
