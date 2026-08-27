"""Tests migrados a applications/backend/tests/test_api.py."""

import pytest

pytest.skip(
    "Los tests de la API viven en applications/backend/tests/",
    allow_module_level=True,
)
