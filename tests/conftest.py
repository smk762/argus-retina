"""Tests never read a developer's suite file (argus_cortex.settings): set before any module reads a setting."""

import os

os.environ["ARGUS_ENV_FILE"] = "off"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _admin_routes_open(monkeypatch):
    """The admin routes are closed without ARGUS_ADMIN_TOKEN. Tests that aren't about that open them, as a developer
    would; the tests of the gate take the variable away again."""
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ARGUS_ADMIN_INSECURE", "1")
