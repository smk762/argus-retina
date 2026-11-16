"""The OCR image's shaping of RapidOCR output: boxes, reading order, and the contract's ``text`` field."""

from __future__ import annotations

from types import SimpleNamespace

from argus_retina.ocr import reading_order, to_box, to_lines


def _line(text, x, y, w=40, h=10, score=0.9):
    return {"class": "text", "score": score, "box": [x, y, w, h], "text": text}


def test_a_quad_becomes_an_axis_aligned_box():
    assert to_box([[10.4, 22.0], [60.9, 20.5], [61.0, 35.2], [9.8, 36.0]]) == [9, 20, 52, 16]


def test_reading_order_is_rows_top_to_bottom_then_left_to_right():
    # "B" sits slightly lower than "A" but on the same row; "C" is the next row.
    lines = [_line("C", 0, 40), _line("B", 100, 3), _line("A", 0, 0)]
    assert [d["text"] for d in reading_order(lines)] == ["A", "B", "C"]
    assert reading_order([]) == []


def test_rapidocr_output_becomes_nudenet_shaped_text_lines():
    quad = [[0, 0], [50, 0], [50, 12], [0, 12]]
    lower = [[0, 30], [50, 30], [50, 42], [0, 42]]
    result = SimpleNamespace(
        txts=("Stockholm", "Henri Bequerel", "  "), scores=(0.97, 0.91, 0.6), boxes=(lower, quad, quad)
    )
    assert to_lines(result) == [
        {"class": "text", "score": 0.91, "box": [0, 0, 50, 12], "text": "Henri Bequerel"},
        {"class": "text", "score": 0.97, "box": [0, 30, 50, 12], "text": "Stockholm"},
    ]


def test_an_image_without_text_has_no_lines():
    assert to_lines(SimpleNamespace(txts=None, scores=None, boxes=None)) == []
