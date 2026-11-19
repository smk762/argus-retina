"""Watermark detector (YOLO, CPU): ``uvicorn argus_retina.watermark:app``. Serves curator's ``crops.detect``.

The image ships **without weights**. The deployer mounts them at ``WATERMARK_MODEL``. The model the suite runs is
fancyfeast's YOLO11x, behind JoyCaption's watermark tags: it has no stated licence, so it is never baked into a
published image (`make watermark-weights` fetches it locally, pinned). mnemic's CC-BY-4.0 YOLOv8s was tried in its
place and caught 15 of 32 watermarked images against the YOLO11x's 28 at curator's 0.4 cutoff (argus-cortex#88).
Without a file at ``WATERMARK_MODEL``, ``/health`` reports ``error`` and says why.

Settings: ``WATERMARK_MODEL`` (default ``/models/watermark.pt``), ``WATERMARK_CONF`` (0.2, a reporting floor: curator
applies its own ``min_conf`` of 0.4), ``WATERMARK_IOU`` (0.5), ``WATERMARK_IMGSZ`` (1024, the YOLO11x's training
size). The single class is ``watermark``. Moved from argus-cull's ``watermark/`` sidecar.
"""

from __future__ import annotations

import os
from pathlib import Path

from argus_retina.sidecar import Detect, Resident, configure_logging, create_app, decode_bgr
from argus_retina.yolo import boxes

MODEL = Path(os.environ.get("WATERMARK_MODEL", "/models/watermark.pt"))
CONF = float(os.environ.get("WATERMARK_CONF", "0.2") or "0.2")
IOU = float(os.environ.get("WATERMARK_IOU", "0.5") or "0.5")
IMGSZ = int(os.environ.get("WATERMARK_IMGSZ", "1024") or "1024")


def load() -> Detect:
    if not MODEL.is_file():
        raise FileNotFoundError(f"watermark model not found: {MODEL} (mount the weights there; see the README)")
    # Only in the watermark image, which serves it over HTTP; this repo is AGPL-3.0 (cortex#88).
    from ultralytics import YOLO  # noqa: PLC0415  # nosemgrep: agpl-import - an AGPL sidecar

    model = YOLO(str(MODEL))

    def detect_all(images: list[bytes]) -> list[list[dict]]:
        return [
            boxes(model.predict(decode_bgr(data), conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)) for data in images
        ]

    return detect_all


resident = Resident("Watermark", load, {"model": MODEL.name, "conf": CONF, "iou": IOU, "imgsz": IMGSZ})
configure_logging()
app = create_app(resident)
