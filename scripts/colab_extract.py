"""Extrae el ZIP del proyecto y normaliza la ruta raíz para Colab."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_NAME = "metricas-proyecto-ml"
DEFAULT_ROOT = Path("/content") / PROJECT_NAME


def _is_project_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "requirements.txt").is_file()
        and (path / "data" / "processed" / "train.csv").is_file()
    )


def find_project_root(search_dir: Path) -> Path | None:
    if _is_project_root(search_dir):
        return search_dir
    nested = search_dir / PROJECT_NAME
    if _is_project_root(nested):
        return nested
    for child in search_dir.iterdir():
        if child.is_dir() and _is_project_root(child):
            return child
    return None


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                target_dir = extract_dir / name.rstrip("/")
                if name:
                    target_dir.mkdir(parents=True, exist_ok=True)
                continue
            target = extract_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())


def normalize_project_root(found: Path, target: Path = DEFAULT_ROOT) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if found.resolve() == target.resolve():
        return target
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(found), str(target))
    return target


def setup_from_zip(zip_path: Path, target: Path = DEFAULT_ROOT) -> Path:
    extract_dir = target.parent / "_zip_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_zip(zip_path, extract_dir)
    found = find_project_root(extract_dir)
    if found is None:
        found = find_project_root(extract_dir.parent)
    if found is None:
        raise FileNotFoundError(
            f"No se encontró la raíz del proyecto tras extraer {zip_path}. "
            "Regenera el ZIP con: python scripts/build_colab_zip.py"
        )

    root = normalize_project_root(found, target)
    shutil.rmtree(extract_dir, ignore_errors=True)
    return root
