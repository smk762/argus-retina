"""argus-retina — Shared detector servers for the Argus suite: NudeNet, EraX, watermark, OCR, guards, faces, synthetic scores and C2PA manifests behind one HTTP contract"""

from __future__ import annotations

try:
    # Written by hatch-vcs at build time (see pyproject [tool.hatch.build.hooks.vcs]).
    from argus_retina._version import __version__
except ImportError:  # running from a source checkout that hasn't been built
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("argus-retina")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
