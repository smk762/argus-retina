from __future__ import annotations

import argus_retina


def test_version_is_exposed() -> None:
    assert isinstance(argus_retina.__version__, str)
    assert argus_retina.__version__
