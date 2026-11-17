"""The contract every image serves, around a stand-in detector (no model, no network)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading

import cv2
import numpy as np
import pytest
from argus_cortex.gpu_wire import ResidentHealth
from argus_cortex.moderation.nudenet import NudeNetInfo
from fastapi.testclient import TestClient

import argus_retina
from argus_retina import sidecar
from argus_retina.sidecar import Resident, UnreadableImage, create_app, decode_bgr


def png(gray: bool = False) -> bytes:
    img = np.zeros((8, 6) if gray else (8, 6, 3), np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def fake_loader(calls: list[int]):
    def load():
        def detect_all(images: list[bytes]) -> list[list[dict]]:
            calls.append(len(images))
            out = []
            for n, data in enumerate(images):
                h, w = decode_bgr(data).shape[:2]
                out.append([{"class": "FACE_FEMALE", "score": 0.5 + n / 10, "box": [0, 0, w, h]}])
            return out

        return detect_all

    return load


def client(loader, info=None) -> tuple[TestClient, Resident]:
    r = Resident("NudeNet", loader, info or {"model": "640m.onnx", "resolution": 640})
    return TestClient(create_app(r, load=False)), r


def test_infer_answers_each_image_in_field_order():
    calls: list[int] = []
    c, r = client(fake_loader(calls))
    r.load()
    files = [("f1", ("a.png", png(), "image/png")), ("f2", ("b.png", png(gray=True), "image/png"))]
    body = c.post("/infer", files=files).json()
    assert body["success"] is True and calls == [2]
    assert [p[0]["score"] for p in body["prediction"]] == [0.5, 0.6]
    assert body["prediction"][1][0]["box"] == [0, 0, 6, 8]  # a grayscale frame decodes as three channels


def test_infer_refuses_no_images_and_unreadable_ones():
    c, r = client(fake_loader([]))
    r.load()
    assert c.post("/infer", data={"x": "1"}).status_code == 400
    bad = c.post("/infer", files=[("f1", ("a.png", b"not an image", "image/png"))])
    assert bad.status_code == 422


@pytest.mark.parametrize("data", [b"", png()[:40]], ids=["empty", "truncated"])
def test_an_empty_or_broken_part_is_unreadable_not_a_cv2_error(data):
    with pytest.raises(UnreadableImage):
        decode_bgr(data)


def test_an_empty_part_is_a_422():
    c, r = client(fake_loader([]))
    r.load()
    assert c.post("/infer", files=[("f1", ("a.png", b"", "image/png"))]).status_code == 422


def test_opencv_refuses_images_over_the_pixel_budget_before_decoding():
    # OpenCV reads the limit when cv2 is imported, so this runs in a fresh interpreter, importing the sidecar first
    # as every image module does.
    assert os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] == str(sidecar.MAX_PIXELS) and sidecar.MAX_PIXELS <= 2**27
    code = textwrap.dedent(
        """
        from argus_retina.sidecar import UnreadableImage, decode_bgr
        import cv2, numpy as np
        ok, buf = cv2.imencode(".png", np.zeros((40, 40, 3), np.uint8))
        assert decode_bgr(cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))[1].tobytes()).shape == (20, 20, 3)
        try:
            decode_bgr(buf.tobytes())
        except UnreadableImage:
            print("refused")
        """
    )
    env = {**os.environ, "ARGUS_RETINA_MAX_PIXELS": "1000"}
    env.pop("OPENCV_IO_MAX_IMAGE_PIXELS", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "refused"


def test_a_decoded_image_over_the_budget_is_unreadable(monkeypatch):
    # The fallback when cv2 was imported before the sidecar (as here) and so never saw the limit.
    monkeypatch.setattr(sidecar, "MAX_PIXELS", 47)
    with pytest.raises(UnreadableImage, match="over 47 pixels"):
        decode_bgr(png())  # 8 x 6 = 48


@pytest.mark.parametrize("exc", [RuntimeError, ValueError])
def test_a_detector_failure_is_a_500(exc):
    # A ValueError inside the model is the server's fault too: only an unreadable image is a 422.
    def load():
        def detect_all(images):
            raise exc("onnx exploded")

        return detect_all

    c, r = client(load)
    r.load()
    r_ = c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))])
    assert r_.status_code == 500 and "onnx" not in r_.text  # the detail stays in the log


def test_health_is_cortex_resident_health_and_names_the_model(monkeypatch):
    monkeypatch.setenv("ARGUS_RETINA_COMMIT", "abc123")
    c, r = client(fake_loader([]))
    before = c.get("/health").json()
    assert before["model_state"] == "unloaded" and before["idle_s"] is None
    assert c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))]).status_code == 503

    r.load()
    body = c.get("/health").json()
    h = ResidentHealth.model_validate(body)
    assert h.model_state == "cpu" and h.busy is False and h.idle_s is not None
    assert body["commit"] == "abc123" and body["version"] == argus_retina.__version__  # AGPL section 13
    # what argus_cortex's NudeNetClient.info() reads from it
    assert NudeNetInfo(model=body["model"], resolution=body["resolution"]).detector == "640m"


def test_a_model_that_fails_to_load_says_so():
    def load():
        raise FileNotFoundError("NudeNet model not found: /models/640m.onnx")

    c, r = client(load)
    r.load()
    body = c.get("/health").json()
    assert ResidentHealth.model_validate(body).model_state == "error"
    assert "not found" in body["error"]
    assert c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))]).status_code == 503


def test_busy_while_detecting_and_one_at_a_time():
    started, release = threading.Event(), threading.Event()
    running = []

    def load():
        def detect_all(images):
            running.append(1)
            assert len(running) == 1  # never two at once
            started.set()
            release.wait(5)
            running.pop()
            return [[] for _ in images]

        return detect_all

    _, r = client(load)
    r.load()
    t = threading.Thread(target=r.detect, args=([b"x"],))
    t.start()
    assert started.wait(5)
    h = r.health()
    assert h["busy"] is True and h["idle_s"] is None
    t2 = threading.Thread(target=r.detect, args=([b"y"],))
    t2.start()
    release.set()
    t.join(5)
    t2.join(5)
    assert r.health()["busy"] is False


def test_the_app_loads_its_model_at_startup():
    loaded = threading.Event()

    def load():
        loaded.set()
        return lambda images: [[] for _ in images]

    r = Resident("NudeNet", load, {"model": "640m.onnx"})
    with TestClient(create_app(r)):
        assert loaded.wait(5)


@pytest.mark.parametrize(
    "module", ["argus_retina.nudenet", "argus_retina.erax", "argus_retina.watermark", "argus_retina.ocr"]
)
def test_the_image_modules_import_without_their_detector(module, monkeypatch, tmp_path):
    # nudenet / ultralytics are only in their images: importing the module must not need them
    import importlib

    mod = importlib.import_module(module)
    assert mod.resident.state == "unloaded"
    monkeypatch.setattr(mod, "DET" if module.endswith(".ocr") else "MODEL", tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError):
        mod.load()
