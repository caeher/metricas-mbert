"""Tests de rutas del proyecto."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.paths import CORPUS_CSV, PROJECT_ROOT, TEST_CSV, TRAIN_CSV


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_data_files_exist():
    assert TRAIN_CSV.exists()
    assert TEST_CSV.exists()
    assert CORPUS_CSV.exists()
