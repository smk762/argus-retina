"""EraX-NSFW V1.0 (YOLO11s, CPU): ``uvicorn argus_retina.erax:app``.

A second nudity detector beside NudeNet, behind the same ``POST /infer`` shape, so a caller reuses one client
path. Classes are EraX's own: ``anus``, ``make_love``, ``nipple``, ``penis``, ``vagina``.

Settings: ``ERAX_MODEL`` (default ``/models/erax_nsfw_yolo11s.pt``), ``ERAX_CONF`` (0.2), ``ERAX_IOU`` (0.3).
The weights are Apache-2.0; Ultralytics is AGPL-3.0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from argus_retina.sidecar import Detect, Resident, configure_logging, create_app, decode_bgr

MODEL = Path(os.environ.get("ERAX_MODEL", "/models/erax_nsfw_yolo11s.pt"))
CONF = float(os.environ.get("ERAX_CONF", "0.2") or "0.2")
IOU = float(os.environ.get("ERAX_IOU", "0.3") or "0.3")


def boxes(results: Any) -> list[dict[str, Any]]:
    """Ultralytics results as NudeNet-shaped detections: ``{"class", "score", "box": [x, y, w, h]}``."""
    out: list[dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(
                {
                    "class": result.names[int(box.cls[0])],
                    "score": float(box.conf[0]),
                    "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                }
            )
    return out


def load() -> Detect:
    if not MODEL.is_file():
        raise FileNotFoundError(f"EraX model not found: {MODEL}")
    from ultralytics import YOLO  # noqa: PLC0415 - only in the erax image

    model = YOLO(str(MODEL))

    def detect_all(images: list[bytes]) -> list[list[dict]]:
        return [boxes(model.predict(decode_bgr(data), conf=CONF, iou=IOU, verbose=False)) for data in images]

    return detect_all


resident = Resident("EraX", load, {"model": MODEL.name, "conf": CONF, "iou": IOU})
configure_logging()
app = create_app(resident)
