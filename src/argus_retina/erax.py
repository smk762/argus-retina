"""EraX-NSFW V1.0 (YOLO11s, CPU): ``uvicorn argus_retina.erax:app``.

A second nudity detector beside NudeNet, behind the same ``POST /infer`` shape, so a caller reuses one client
path. Classes are EraX's own: ``anus``, ``make_love``, ``nipple``, ``penis``, ``vagina``.

Settings: ``ERAX_MODEL`` (default ``/models/erax_nsfw_yolo11s.pt``), ``ERAX_CONF`` (0.2), ``ERAX_IOU`` (0.3).
The weights are Apache-2.0; Ultralytics is AGPL-3.0.
"""

from __future__ import annotations

import os
from pathlib import Path

from argus_retina.sidecar import Detect, Resident, configure_logging, create_app, decode_bgr
from argus_retina.yolo import boxes

MODEL = Path(os.environ.get("ERAX_MODEL", "/models/erax_nsfw_yolo11s.pt"))
CONF = float(os.environ.get("ERAX_CONF", "0.2") or "0.2")
IOU = float(os.environ.get("ERAX_IOU", "0.3") or "0.3")


def load() -> Detect:
    if not MODEL.is_file():
        raise FileNotFoundError(f"EraX model not found: {MODEL}")
    # Only in the erax image, which serves it over HTTP; this repo is AGPL-3.0 (cortex#88).
    from ultralytics import YOLO  # noqa: PLC0415  # nosemgrep: agpl-import - an AGPL sidecar

    model = YOLO(str(MODEL))

    def detect_all(images: list[bytes]) -> list[list[dict]]:
        return [boxes(model.predict(decode_bgr(data), conf=CONF, iou=IOU, verbose=False)) for data in images]

    return detect_all


resident = Resident("EraX", load, {"model": MODEL.name, "conf": CONF, "iou": IOU})
configure_logging()
app = create_app(resident)
