"""Tests de mapeo de etiquetas externas."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.benchmark import (
    _map_alia,
    _map_exist,
    _map_piuba,
    _map_sinodio,
    _map_textdetox,
    project_prediction,
)


def test_map_piuba_no_hate():
    assert _map_piuba({"HATEFUL": 0}) == 0


def test_map_piuba_hate_no_calls():
    assert _map_piuba({"HATEFUL": 1, "CALLS": 0}) == 2


def test_map_piuba_threat():
    assert _map_piuba({"HATEFUL": 1, "CALLS": 1}) == 3


def test_map_sinodio():
    assert _map_sinodio({"label": 0}) == 0
    assert _map_sinodio({"label": 1}) == 2
    assert _map_sinodio({"label": "no_hate"}) == 0
    assert _map_sinodio({"label": "hate_speech"}) == 2


def test_map_textdetox():
    assert _map_textdetox({"toxic": 0}) == 0
    assert _map_textdetox({"toxic": 1}) == 1


def test_map_exist():
    assert _map_exist({"label": "non-sexist"}) == 0
    assert _map_exist({"label": "sexist"}) == 2


def test_map_alia():
    assert _map_alia({"foe_class": 0}) == 0
    assert _map_alia({"foe_class": 1}) == 2


def test_project_prediction():
    assert project_prediction(0)["binary_toxic"] == 0
    assert project_prediction(2)["binary_toxic"] == 1
