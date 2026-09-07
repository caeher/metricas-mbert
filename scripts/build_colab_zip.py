#!/usr/bin/env python
"""Genera metricas-proyecto-ml-colab.zip compatible con Python zipfile en Linux/Colab."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_NAME = "metricas-proyecto-ml"
ZIP_NAME = "metricas-proyecto-ml-colab.zip"
SKIP_NOTEBOOK_SUFFIX = "_executed.ipynb"
SKIP_PATH_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
MBERT_SOURCE_DIR = "mbert-sv-b2"
MBERT_DEST_DIR = "mbert-sv"
REQUIRED_MBERT_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")


def _copy_tree(src: Path, dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _stage_mbert_weights(repo_root: Path, staging: Path) -> None:
    """Copia pesos finales de mbert-sv-b2 → models/mbert-sv (sin checkpoints intermedios)."""
    source = repo_root / "models" / MBERT_SOURCE_DIR
    dest = staging / "models" / MBERT_DEST_DIR

    if not source.is_dir():
        raise FileNotFoundError(
            f"No se encontró {source}. Entrena o restaura el checkpoint antes de generar el ZIP."
        )

    missing = [name for name in REQUIRED_MBERT_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Faltan archivos requeridos en {source}: {', '.join(missing)}. "
            "No se generará un ZIP incompleto."
        )

    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_dir():
            if item.name.startswith("checkpoint-"):
                continue
            _copy_tree(item, dest / item.name)
            continue
        shutil.copy2(item, dest / item.name)


def _stage_baselines(repo_root: Path, staging: Path) -> None:
    """Copia baselines TF-IDF pre-entrenados."""
    models_dir = repo_root / "models"
    dest_models = staging / "models"
    dest_models.mkdir(parents=True, exist_ok=True)

    baselines = sorted(models_dir.glob("baseline_*.joblib"))
    if not baselines:
        raise FileNotFoundError(
            f"No se encontraron baselines en {models_dir}. "
            "Ejecuta el notebook 03 o scripts de baselines antes de generar el ZIP."
        )

    for baseline in baselines:
        shutil.copy2(baseline, dest_models / baseline.name)


def _stage_project(repo_root: Path, staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    _copy_tree(repo_root / "data" / "processed", staging / "data" / "processed")
    _copy_tree(repo_root / "config", staging / "config")

    src_pkg = repo_root / "src" / "discurso_odio"
    dest_pkg = staging / "src" / "discurso_odio"
    dest_pkg.mkdir(parents=True, exist_ok=True)
    for file in src_pkg.rglob("*"):
        if not file.is_file():
            continue
        if any(part in SKIP_PATH_PARTS for part in file.parts):
            continue
        if file.suffix == ".pyc":
            continue
        rel = file.relative_to(src_pkg)
        target = dest_pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)

    for name in ("requirements.txt", "requirements-colab.txt", "pyproject.toml", ".env.example", "README.md"):
        _copy_tree(repo_root / name, staging / name)

    nb_dest = staging / "notebooks"
    nb_dest.mkdir(parents=True, exist_ok=True)
    for nb in sorted((repo_root / "notebooks").glob("*.ipynb")):
        if nb.name.endswith(SKIP_NOTEBOOK_SUFFIX):
            continue
        shutil.copy2(nb, nb_dest / nb.name)

    _copy_tree(repo_root / "scripts", staging / "scripts")

    for folder in ("reports/figures", "reports/metrics", "reports/tables"):
        (staging / folder).mkdir(parents=True, exist_ok=True)

    _stage_baselines(repo_root, staging)
    _stage_mbert_weights(repo_root, staging)


def _write_zip(staging: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(staging.rglob("*")):
            if file.is_dir():
                continue
            rel = file.relative_to(staging).as_posix()
            arcname = f"{PROJECT_NAME}/{rel}"
            zf.write(file, arcname)


def _copy_setup_notebook(repo_root: Path, colab_dir: Path) -> None:
    """Copia el notebook de entrada a google-colab/ para acceso directo."""
    setup_nb = repo_root / "notebooks" / "00_colab_setup.ipynb"
    if not setup_nb.is_file():
        raise FileNotFoundError(f"No se encontró {setup_nb}")
    colab_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(setup_nb, colab_dir / "00_colab_setup.ipynb")


def build_colab_zip(repo_root: Path | None = None, out_zip: Path | None = None) -> Path:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    colab_dir = repo_root / "google-colab"
    out_zip = out_zip or colab_dir / ZIP_NAME
    staging = colab_dir / PROJECT_NAME

    _stage_project(repo_root, staging)
    _write_zip(staging, out_zip)
    _copy_setup_notebook(repo_root, colab_dir)

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"ZIP creado: {out_zip} ({size_mb:.2f} MB)")
    print(f"Staging: {staging}")
    print(f"Notebook setup: {colab_dir / '00_colab_setup.ipynb'}")
    print(
        f"Estructura: {PROJECT_NAME}/data, {PROJECT_NAME}/notebooks, "
        f"{PROJECT_NAME}/models/mbert-sv, ..."
    )
    return out_zip


if __name__ == "__main__":
    build_colab_zip()
