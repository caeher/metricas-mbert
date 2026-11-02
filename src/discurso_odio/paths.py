"""Rutas centralizadas del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
CANDIDATES_DIR = PROJECT_ROOT / "data" / "sources" / "candidates"
VERSIONES_DIR = PROJECT_ROOT / "versiones"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
REPORTS_METRICS = PROJECT_ROOT / "reports" / "metrics"
REPORTS_TABLES = PROJECT_ROOT / "reports" / "tables"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

TRAIN_CSV = DATA_PROCESSED / "train.csv"
VAL_CSV = DATA_PROCESSED / "val.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
TEST_ADVERSARIAL_CSV = DATA_PROCESSED / "test_adversarial.csv"
TEST_ADVERSARIAL_144_CSV = DATA_PROCESSED / "test_adversarial_144.csv"
DATASET_METADATA_JSON = DATA_PROCESSED / "metadata.json"
RELEASES_DIR = PROJECT_ROOT / "data" / "releases"
CORPUS_CSV = DATA_PROCESSED / "corpus_limpio.csv"
CORPUS_COMPLETO_CSV = DATA_INTERIM / "corpus_completo.csv"
WILLIAN_FACEBOOK_CSV = VERSIONES_DIR / "v3" / "sources" / "dataset_willian_facebook.csv"
LENGUAJE_OFENSIVO_CSV = CANDIDATES_DIR / "lenguaje_ofensivo.csv"
THENEWDATA_X_CSV = CANDIDATES_DIR / "thenewdata_X.csv"
ADVERSARIAL_BATTERY_MANIFEST_PATH = REPORTS_METRICS / "adversarial_battery_manifest.json"
ADVERSARIAL_144_BATTERY_MANIFEST_PATH = REPORTS_METRICS / "adversarial_144_battery_manifest.json"

for _path in (
    DATA_PROCESSED,
    DATA_EXTERNAL,
    DATA_INTERIM,
    CANDIDATES_DIR,
    MODELS_DIR,
    RELEASES_DIR,
    VERSIONES_DIR,
    REPORTS_FIGURES,
    REPORTS_METRICS,
    REPORTS_TABLES,
):
    _path.mkdir(parents=True, exist_ok=True)


def version_model_dir(version_id: str) -> Path:
    """Checkpoint HuggingFace bajo versiones/vN/model/."""
    return VERSIONES_DIR / version_id / "model"
