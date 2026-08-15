"""Prueba extracción del ZIP como en Colab."""

import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.colab_extract import find_project_root, setup_from_zip

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "metricas-proyecto-ml-colab.zip"
BUILD_SCRIPT = ROOT / "scripts" / "build_colab_zip.py"
PROJECT_NAME = "metricas-proyecto-ml"


@pytest.fixture(scope="module")
def colab_zip(tmp_path_factory) -> Path:
    if not ZIP_PATH.exists():
        subprocess.run(["python", str(BUILD_SCRIPT)], check=True, cwd=ROOT)
    return ZIP_PATH


def test_colab_zip_has_project_root_folder(colab_zip: Path):
    with zipfile.ZipFile(colab_zip) as zf:
        names = zf.namelist()
    assert any(name.startswith(f"{PROJECT_NAME}/") for name in names)
    assert not any(name == "data/processed/train.csv" for name in names)


def test_colab_zip_contains_required_files(colab_zip: Path):
    required = [
        f"{PROJECT_NAME}/data/processed/train.csv",
        f"{PROJECT_NAME}/pyproject.toml",
        f"{PROJECT_NAME}/requirements.txt",
        f"{PROJECT_NAME}/notebooks/00_colab_setup.ipynb",
    ]
    with zipfile.ZipFile(colab_zip) as zf:
        names = set(zf.namelist())
    for entry in required:
        assert entry in names, f"Falta en ZIP: {entry}"


def test_setup_from_zip_extracts_to_target(tmp_path: Path, colab_zip: Path):
    target = tmp_path / PROJECT_NAME
    root = setup_from_zip(colab_zip, target=target)
    assert root == target
    assert (root / "pyproject.toml").is_file()
    assert (root / "data" / "processed" / "train.csv").is_file()
    assert find_project_root(root) == root
