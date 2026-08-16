"""Tests de consolidación de cortes oficiales mBERT (T-05)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.official_mbert import (
    MBERT_OFFICIAL_RUN_PATH,
    build_platform_traceability_rows,
)
from discurso_odio.evaluation.slices import MIN_SLICE_N_FOR_CI
from discurso_odio.paths import REPORTS_METRICS

SLICE_METRICS_PATH = REPORTS_METRICS / "slice_metrics.csv"
DIALECT_BIAS_PATH = REPORTS_METRICS / "dialect_bias_probe.csv"
PLATFORM_TRACE_PATH = REPORTS_METRICS / "mbert_platform_traceability.csv"
PERCLASS_PATH = REPORTS_METRICS / "slice_perclass_metrics.csv"


@pytest.mark.skipif(not SLICE_METRICS_PATH.exists(), reason="Falta slice_metrics.csv")
def test_mbert_slices_share_official_run_id():
    with open(MBERT_OFFICIAL_RUN_PATH, encoding="utf-8") as handle:
        official = json.load(handle)
    df = pd.read_csv(SLICE_METRICS_PATH)
    mbert = df[df["model"] == "mbert"]
    assert not mbert.empty
    assert set(mbert["run_id"].unique()) == {official["run_id"]}
    assert all(str(c) == official["checkpoint"] for c in mbert["checkpoint"].unique())


@pytest.mark.skipif(not SLICE_METRICS_PATH.exists(), reason="Falta slice_metrics.csv")
def test_platform_slices_have_distribution_and_metrics():
    df = pd.read_csv(SLICE_METRICS_PATH)
    mbert = df[df["model"] == "mbert"]
    for slice_name in ("plataforma_X", "plataforma_Facebook"):
        row = mbert[mbert["slice"] == slice_name].iloc[0]
        assert int(row["n"]) > 0
        assert pd.notna(row["f1_macro"])
        assert pd.notna(row["accuracy"])
        assert int(row["n0"]) + int(row["n1"]) + int(row["n2"]) + int(row["n3"]) == int(row["n"])
        assert row["definition"]


@pytest.mark.skipif(not SLICE_METRICS_PATH.exists(), reason="Falta slice_metrics.csv")
def test_indicator_slices_use_si_no_groups():
    df = pd.read_csv(SLICE_METRICS_PATH)
    mbert = df[df["model"] == "mbert"]
    for prefix in ("sarcasmo", "ironia", "contexto_politico"):
        si = mbert[mbert["slice"] == f"{prefix}_si"]
        no = mbert[mbert["slice"] == f"{prefix}_no"]
        assert not si.empty and not no.empty
        assert int(si.iloc[0]["n"]) > 0
        assert int(no.iloc[0]["n"]) > 0


@pytest.mark.skipif(not SLICE_METRICS_PATH.exists(), reason="Falta slice_metrics.csv")
def test_small_slice_has_low_power_flag():
    df = pd.read_csv(SLICE_METRICS_PATH)
    mbert = df[df["model"] == "mbert"]
    ironia_si = mbert[mbert["slice"] == "ironia_si"].iloc[0]
    assert int(ironia_si["n"]) < MIN_SLICE_N_FOR_CI or bool(ironia_si["low_power"])
    assert "n=" in str(ironia_si["power_note"]) or "baja potencia" in str(ironia_si["power_note"])


def test_platform_traceability_marks_historical_not_reproducible():
    rows = build_platform_traceability_rows()
    historical = [r for r in rows if r["status"] == "not_reproducible"]
    assert len(historical) == 2
    platforms = {r["platform"] for r in historical}
    assert platforms == {"X", "Facebook"}


@pytest.mark.skipif(not DIALECT_BIAS_PATH.exists(), reason="Falta dialect_bias_probe.csv")
def test_dialect_bias_shares_official_run_id():
    with open(MBERT_OFFICIAL_RUN_PATH, encoding="utf-8") as handle:
        official = json.load(handle)
    df = pd.read_csv(DIALECT_BIAS_PATH)
    mbert = df[df["model"] == "mbert"]
    assert set(mbert["run_id"].unique()) == {official["run_id"]}


@pytest.mark.skipif(not PERCLASS_PATH.exists(), reason="Falta slice_perclass_metrics.csv")
def test_perclass_metrics_only_for_large_slices():
    slices = pd.read_csv(SLICE_METRICS_PATH)
    perclass = pd.read_csv(PERCLASS_PATH)
    mbert_slices = set(slices[slices["model"] == "mbert"]["slice"])
    perclass_slices = set(perclass[perclass["model"] == "mbert"]["slice"])
    for slice_name in perclass_slices:
        n = int(slices[(slices["model"] == "mbert") & (slices["slice"] == slice_name)].iloc[0]["n"])
        assert n >= MIN_SLICE_N_FOR_CI
    assert "ironia_si" not in perclass_slices or int(
        slices[(slices["model"] == "mbert") & (slices["slice"] == "ironia_si")].iloc[0]["n"]
    ) >= MIN_SLICE_N_FOR_CI
