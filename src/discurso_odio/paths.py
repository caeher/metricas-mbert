"""Rutas centralizadas del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
REPORTS_METRICS = PROJECT_ROOT / "reports" / "metrics"
REPORTS_TABLES = PROJECT_ROOT / "reports" / "tables"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

TRAIN_CSV = DATA_PROCESSED / "train.csv"
VAL_CSV = DATA_PROCESSED / "val.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
CORPUS_CSV = DATA_PROCESSED / "corpus_limpio.csv"

for _path in (DATA_PROCESSED, DATA_EXTERNAL, MODELS_DIR, REPORTS_FIGURES, REPORTS_METRICS, REPORTS_TABLES):
    _path.mkdir(parents=True, exist_ok=True)
