#!/usr/bin/env python
"""Genera metricas-proyecto-ml-colab.zip compatible con Python zipfile en Linux/Colab."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_NAME = "metricas-proyecto-ml"
SKIP_NOTEBOOK_SUFFIX = "_executed.ipynb"
SKIP_PATH_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _copy_tree(src: Path, dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


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

    for folder in ("models", "reports/figures", "reports/metrics", "reports/tables"):
        (staging / folder).mkdir(parents=True, exist_ok=True)


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


def build_colab_zip(repo_root: Path | None = None, out_zip: Path | None = None) -> Path:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    out_zip = out_zip or repo_root / "metricas-proyecto-ml-colab.zip"
    staging = repo_root / ".colab_zip_staging" / PROJECT_NAME

    _stage_project(repo_root, staging)
    _write_zip(staging, out_zip)
    shutil.rmtree(staging.parent, ignore_errors=True)

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"ZIP creado: {out_zip} ({size_mb:.2f} MB)")
    print(f"Estructura: {PROJECT_NAME}/data, {PROJECT_NAME}/notebooks, ...")
    return out_zip


if __name__ == "__main__":
    build_colab_zip()
