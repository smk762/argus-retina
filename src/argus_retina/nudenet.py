"""NudeNet with its 640m detector: ``uvicorn argus_retina.nudenet:app``.

The upstream ``ghcr.io/notai-tech/nudenet`` image and the ``nudenet`` wheel ship only 320n, which scores
differently (and has a ``y``-offset bug in the upstream server). This runs
``NudeDetector(model_path=640m.onnx, inference_resolution=640)`` behind the same ``POST /infer``.
``/health`` names the weights file and resolution, which ``NudeNetClient.info()`` reads as ``detector: 640m``.

Settings: ``NUDENET_MODEL`` (default ``/models/640m.onnx``), ``NUDENET_RESOLUTION`` (default 640).
NudeNet is AGPL-3.0; callers only ever talk HTTP to it.
"""

from __future__ import annotations

import os
from pathlib import Path

from argus_retina.sidecar import Detect, Resident, configure_logging, create_app, decode_bgr

MODEL = Path(os.environ.get("NUDENET_MODEL", "/models/640m.onnx"))
RESOLUTION = int(os.environ.get("NUDENET_RESOLUTION", "640") or "640")


def load() -> Detect:
    if not MODEL.is_file():
        raise FileNotFoundError(f"NudeNet model not found: {MODEL}")
    from nudenet import NudeDetector  # noqa: PLC0415 - only in the nudenet image

    detector = NudeDetector(model_path=str(MODEL), inference_resolution=RESOLUTION)

    def detect_all(images: list[bytes]) -> list[list[dict]]:
        # An array, not bytes: NudeNet's own bytes path uses IMREAD_UNCHANGED, which fails on grayscale frames.
        return [detector.detect(decode_bgr(data)) for data in images]

    return detect_all


resident = Resident("NudeNet", load, {"model": MODEL.name, "resolution": RESOLUTION})
configure_logging()
app = create_app(resident)
