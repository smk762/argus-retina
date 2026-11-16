"""Ultralytics results in NudeNet's detection shape, for the YOLO images (EraX, watermark)."""

from __future__ import annotations

from typing import Any


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
