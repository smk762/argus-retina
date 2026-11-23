"""Tests never read a developer's suite file (argus_cortex.settings): set before any module reads a setting."""

import os

os.environ["ARGUS_ENV_FILE"] = "off"
