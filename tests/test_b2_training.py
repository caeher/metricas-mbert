"""Tests de dataset B2 y receta de entrenamiento mBERT."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import (
    THREAT_SYNTHETIC_ORIGINS,
    compute_sample_loss_weights,
    get_b2_class_weights,
    load_b2_metadata,
    load_split,
    load_train_excluding_adversarial_overlap,
)
from discurso_odio.data.adversarial_battery import BATTERY_144_N, FAMILY_SPECS_144
from discurso_odio.evaluation.adversarial import assert_adversarial_not_in_splits, find_train_overlaps
from discurso_odio.paths import DATASET_METADATA_JSON, TEST_ADVERSARIAL_144_CSV


def test_b2_metadata_version():
    meta = load_b2_metadata()
    assert meta["version"] == "b2"
    assert meta["totales"]["train"] == 3293


def test_b2_class_weights_from_metadata():
    weights = get_b2_class_weights()
    assert len(weights) == 4
    assert abs(weights[0] - 0.9198) < 1e-4
    assert abs(weights[3] - 1.2233) < 1e-4


def test_train_has_origen_and_synthetic_counts():
    train = load_split("train")
    assert "origen" in train.columns
    assert int((train["origen"] == "real").sum()) == 2144
    threat_n = int(train["origen"].isin(THREAT_SYNTHETIC_ORIGINS).sum())
    assert threat_n == 268


def test_val_test_have_no_synthetics():
    for split in ("val", "test"):
        df = load_split(split)
        if "origen" in df.columns:
            assert (df["origen"] == "real").all()


def test_threat_sample_weights_effective_mass():
    train = load_train_excluding_adversarial_overlap()
    weights = compute_sample_loss_weights(train, threat_synthetic_effective_n=90)
    threat_mask = train["origen"].isin(THREAT_SYNTHETIC_ORIGINS)
    effective = float(weights[threat_mask].sum())
    assert abs(effective - 90.0) < 0.5


def test_adversarial_overlap_exclusion():
    overlaps = find_train_overlaps()
    filtered = load_train_excluding_adversarial_overlap()
    assert len(filtered) == len(load_split("train")) - len(overlaps)


def test_adversarial_144_schema():
    assert TEST_ADVERSARIAL_144_CSV.exists()
    df = load_split("adversarial_144")
    assert len(df) == BATTERY_144_N == 144
    for grupo, spec in FAMILY_SPECS_144.items():
        assert int((df["grupo"] == grupo).sum()) == spec["n"]


def test_adversarial_144_no_val_test_leakage():
    assert_adversarial_not_in_splits(splits=("val", "test"))


def test_metadata_json_exists():
    assert DATASET_METADATA_JSON.exists()
    meta = json.loads(DATASET_METADATA_JSON.read_text(encoding="utf-8"))
    assert "pesos_clase" in meta
