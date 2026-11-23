"""POST /synthetic: Community Forensics' probability, and no label."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus_retina.sidecar import Resident
from argus_retina.synthetic import (
    CROP,
    MEAN,
    SCHEMA,
    STD,
    crop_origin,
    detector_state,
    preprocess,
    resized_size,
    to_scores,
)


def png() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.zeros((8, 6, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_resize_matches_torchvision_shorter_side_and_the_crop_is_centred():
    # 800x1000 (w, h): the short side is the width. 440 * 1000 / 800 truncates to 550.
    assert resized_size(800, 1000) == (440, 550)
    assert resized_size(1000, 800) == (550, 440)
    assert resized_size(100, 100) == (440, 440)
    # (440 - 384) / 2 = 28 exactly.
    assert crop_origin(440, 440) == (28, 28)
    # .5 rounds to even: 28.5 -> 28.
    assert crop_origin(441, 384) == (28, 0)


def test_preprocess_is_the_crop_in_imagenet_normalisation():
    frame = preprocess(np.zeros((8, 6, 3), np.uint8))
    assert frame.shape == (3, CROP, CROP) and frame.dtype == np.float32
    expected = (0.0 - MEAN) / STD
    assert np.allclose(frame, expected[:, None, None])


def test_scores_are_four_decimal_places():
    assert to_scores([0.912345, 0.0]) == [{"synthetic": 0.9123}, {"synthetic": 0.0}]


def test_the_published_state_dict_drops_its_vit_prefix():
    raw = {"vit.head.weight": [1], "vit.norm.weight": [2]}
    assert detector_state(raw) == {"head.weight": [1], "norm.weight": [2]}
    assert detector_state({"head.weight": [1]}) == {"head.weight": [1]}
    assert detector_state({}) == {}


def test_post_synthetic_answers_each_image_and_refuses_a_bad_part(monkeypatch):
    import argus_retina.synthetic as synthetic

    original = synthetic.score
    resident = Resident("Synthetic", lambda: object(), {"model": "commfor-model-384"})
    resident.load()
    monkeypatch.setattr(synthetic, "resident", resident)
    monkeypatch.setattr(synthetic, "score", lambda _model, images: to_scores([0.25 for _ in images]))
    client = TestClient(synthetic.app)
    files = [("f1", ("a.png", png(), "image/png")), ("f2", ("b.png", png(), "image/png"))]
    body = client.post("/synthetic", files=files).json()
    assert body["success"] is True and body["schema"] == SCHEMA
    assert body["scores"] == [{"synthetic": 0.25}, {"synthetic": 0.25}]
    assert client.post("/synthetic", data={"x": "1"}).status_code == 400
    # The real decoder runs for an unreadable part, before any model call.
    monkeypatch.setattr(synthetic, "score", original)
    bad = client.post("/synthetic", files=[("f1", ("a.png", b"not an image", "image/png"))])
    assert bad.status_code == 422


def test_a_missing_checkpoint_is_reported_without_importing_torch(monkeypatch, tmp_path):
    import argus_retina.synthetic as synthetic

    monkeypatch.setattr(synthetic, "MODEL", tmp_path / "missing.safetensors")
    with pytest.raises(FileNotFoundError, match="synthetic model not found"):
        synthetic.load()
