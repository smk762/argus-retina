"""POST /faces: InsightFace's box and embedding, MiVOLO's age, and no policy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus_retina.faces import (
    SCHEMA,
    apply_mivolo,
    cuda_options,
    decode_mivolo,
    eye_mouth_open,
    face_record,
    mivolo_batch,
    xyxy_to_xywh,
)
from argus_retina.sidecar import Resident


def png() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.zeros((8, 6, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_a_box_is_xywh_and_pose_is_pitch_yaw_roll():
    assert xyxy_to_xywh([10, 20, 50, 80]) == [10, 20, 40, 60]
    face = SimpleNamespace(
        bbox=[10, 20, 50.5, 80],
        det_score=0.87555,
        normed_embedding=[0.1, -0.2],
        kps=[[1, 2], [3, 4]],
        pose=[1.234, -5.678, 0.4],
        landmark_3d_68=None,
    )
    rec = face_record(face)
    assert rec["bbox"] == [10.0, 20.0, 40.5, 60.0]
    assert rec["det_score"] == 0.8756
    assert rec["embedding"] == [0.1, -0.2]
    assert rec["pose"] == {"pitch": 1.23, "yaw": -5.68, "roll": 0.4}
    assert rec["age_years"] is None and rec["gender"] is None
    assert rec["eye_open"] is None and rec["landmark_3d_68"] is None


def test_open_eyes_and_a_closed_mouth_are_measurements():
    pts = np.zeros((68, 3), np.float64)
    # Right eye, outer corner first: width 3, both lids 2 away from the corners' line.
    pts[36:42, :2] = [[0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]]
    pts[42:48, :2] = pts[36:42, :2] + [10, 0]
    eye, mouth = eye_mouth_open(pts)
    assert eye == 0.6667 and mouth == 0.0
    assert eye_mouth_open(np.zeros((3, 2))) == (None, None)


def test_mivolo_decodes_gender_and_age_years():
    # Softmax of [0, 100] is female. Age raw 0 is the checkpoint's average, 61.
    gender, score, years = decode_mivolo([[0.0, 100.0, 0.0]])[0]
    assert gender == "female" and score == 1.0 and years == 61.0
    assert decode_mivolo([[5.0, 0.0, -0.5]])[0][0] == "male"
    # raw -0.5 -> 0 years; raw 1 -> 183, clipped to 122.
    assert decode_mivolo([[1.0, 0.0, -0.5]])[0][2] == 0.0
    assert decode_mivolo([[1.0, 0.0, 1.0]])[0][2] == 122.0


def test_a_face_crop_becomes_a_mivolo_batch_and_a_speck_is_skipped():
    bgr = np.zeros((40, 30, 3), np.uint8)
    bgr[5:25, 4:20] = 255
    batch, idx = mivolo_batch(bgr, [[4, 5, 16, 20], [0, 0, 1, 1]])
    assert idx == [0] and batch.shape == (1, 6, 384, 384)
    records = [{"age_years": None, "gender": None, "gender_score": None}, {"age_years": None}]
    apply_mivolo(records, [[0.0, 20.0, 0.0]], idx)
    assert records[0]["gender"] == "female" and records[0]["age_years"] == 61.0
    assert records[1]["age_years"] is None


def test_the_cuda_arena_is_capped():
    opts = cuda_options(1024)
    assert opts["arena_extend_strategy"] == "kSameAsRequested"
    assert opts["gpu_mem_limit"] == str(1024 * 1024 * 1024)


def test_post_faces_answers_each_image_and_refuses_a_bad_part(monkeypatch):
    import argus_retina.faces as faces

    original = faces.detect

    def stub(_models, images):
        return [[{"bbox": [0, 0, 1, 1], "det_score": 0.5, "embedding": [0.25]}] for _ in images]

    resident = Resident("Faces", lambda: object(), {"model": "buffalo_l"}, device="gpu")
    resident.load()
    monkeypatch.setattr(faces, "resident", resident)
    monkeypatch.setattr(faces, "detect", stub)
    client = TestClient(faces.app)
    files = [("f1", ("a.png", png(), "image/png")), ("f2", ("b.png", png(), "image/png"))]
    body = client.post("/faces", files=files).json()
    assert body["success"] is True and body["schema"] == SCHEMA and len(body["faces"]) == 2
    assert body["faces"][0][0]["embedding"] == [0.25]
    assert client.post("/faces", data={"x": "1"}).status_code == 400
    # The real decoder runs for an unreadable part, before any model call.
    monkeypatch.setattr(faces, "detect", original)
    bad = client.post("/faces", files=[("f1", ("a.png", b"not an image", "image/png"))])
    assert bad.status_code == 422


def test_a_missing_pack_is_reported_without_importing_insightface(monkeypatch, tmp_path):
    import argus_retina.faces as faces

    monkeypatch.setattr(faces, "ROOT", tmp_path)
    monkeypatch.setattr(faces, "MIVOLO", tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError, match="InsightFace pack not found"):
        faces.load()
