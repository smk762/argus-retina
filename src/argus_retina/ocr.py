"""Text reader (RapidOCR, PP-OCRv6 small, ONNX on the CPU): ``uvicorn argus_retina.ocr:app``.

Reads the text in an image so a caller can check what a caption claims it says (argus-lens reconcile, #5). Each
detection is one line of text, ``{"class": "text", "score", "box": [x, y, w, h], "text"}``, in reading order: top to
bottom, then left to right within a row. Rows, not columns: two facing pages interleave line by line. ``score`` is
the recogniser's confidence in the line. A line the model can barely read (ornate script, Fraktur) comes back with a
low score, not a cleaned-up guess; lines under ``OCR_MIN_SCORE`` are dropped. The model reads what is printed, so a
misspelling in the image stays misspelt.

Chosen over docTR, EasyOCR and Florence-2 on historical and street photos (#5): it found the most of the checked
phrases for its speed, and never wrote text that isn't there, which Florence-2 did.

Settings: ``OCR_DET_MODEL``, ``OCR_REC_MODEL``, ``OCR_CLS_MODEL`` (the three ONNX files, under ``/models/ocr/``),
``OCR_MIN_SCORE`` (0.5, RapidOCR's own default). The models are Apache-2.0 (PaddleOCR), RapidOCR is Apache-2.0.
"""

from __future__ import annotations

import os
from pathlib import Path
from statistics import median
from typing import Any

from argus_retina.sidecar import Detect, Resident, configure_logging, create_app, decode_bgr

MODELS = Path("/models/ocr")
DET = Path(os.environ.get("OCR_DET_MODEL", str(MODELS / "PP-OCRv6_det_small.onnx")))
REC = Path(os.environ.get("OCR_REC_MODEL", str(MODELS / "PP-OCRv6_rec_small.onnx")))
CLS = Path(os.environ.get("OCR_CLS_MODEL", str(MODELS / "ch_ppocr_mobile_v2.0_cls_mobile.onnx")))
MIN_SCORE = float(os.environ.get("OCR_MIN_SCORE", "0.5") or "0.5")


def to_box(quad: Any) -> list[int]:
    """A text line's four corner points as an axis-aligned ``[x, y, w, h]``."""
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    x, y = int(min(xs)), int(min(ys))
    return [x, y, int(max(xs)) - x, int(max(ys)) - y]


def reading_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top to bottom, then left to right within a row.

    Lines whose vertical centres are within half a typical line height of a row's first line share that row.
    """
    if not lines:
        return []
    half = median(d["box"][3] for d in lines) / 2
    rows: list[list[dict[str, Any]]] = []
    for d in sorted(lines, key=lambda d: d["box"][1] + d["box"][3] / 2):
        centre = d["box"][1] + d["box"][3] / 2
        if rows and abs(centre - (rows[-1][0]["box"][1] + rows[-1][0]["box"][3] / 2)) <= half:
            rows[-1].append(d)
        else:
            rows.append([d])
    return [d for row in rows for d in sorted(row, key=lambda d: d["box"][0])]


def to_lines(result: Any) -> list[dict[str, Any]]:
    """RapidOCR's output as NudeNet-shaped detections with the line's ``text``, in reading order."""
    if result.txts is None:
        return []
    lines = [
        {"class": "text", "score": float(score), "box": to_box(quad), "text": text}
        for text, score, quad in zip(result.txts, result.scores, result.boxes, strict=True)
        if text.strip()
    ]
    return reading_order(lines)


def load() -> Detect:
    for model in (DET, REC, CLS):
        if not model.is_file():
            raise FileNotFoundError(f"OCR model not found: {model}")
    from rapidocr import RapidOCR  # noqa: PLC0415 - only in the ocr image

    # Explicit paths: RapidOCR never downloads a model at runtime.
    engine = RapidOCR(
        params={
            "Det.model_path": str(DET),
            "Rec.model_path": str(REC),
            "Cls.model_path": str(CLS),
            "Global.text_score": MIN_SCORE,
            "Global.log_level": "warning",
        }
    )

    def detect_all(images: list[bytes]) -> list[list[dict]]:
        return [to_lines(engine(decode_bgr(data))) for data in images]

    return detect_all


resident = Resident(
    "OCR",
    load,
    {"model": "PP-OCRv6 small (RapidOCR)", "det": DET.name, "rec": REC.name, "min_score": MIN_SCORE},
)
configure_logging()
app = create_app(resident)
