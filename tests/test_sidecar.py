"""The contract every image serves, around a stand-in detector (no model, no network)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time

import cv2
import numpy as np
import pytest
from argus_cortex.gpu_wire import ResidentActionReply, ResidentBusyReply, ResidentHealth
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
    assert os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] == str(sidecar.MAX_PIXELS)
    code = textwrap.dedent(
        """
        from argus_retina.sidecar import UnreadableImage, decode_bgr
        import cv2, numpy as np
        ok, buf = cv2.imencode(".png", np.zeros((40, 40, 3), np.uint8))
        assert decode_bgr(cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))[1].tobytes()).shape == (20, 20, 3)
        try:
            decode_bgr(buf.tobytes())
        except UnreadableImage as exc:
            print(exc)  # OpenCV's header check, not decode_bgr's fallback ("image over ...")
        """
    )
    env = {**os.environ, "ARGUS_RETINA_MAX_PIXELS": "1000"}
    env.pop("OPENCV_IO_MAX_IMAGE_PIXELS", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "unreadable image (or over 1000 pixels)"


@pytest.mark.parametrize("value", ["0", "-5", str(2**64)])
def test_a_pixel_budget_opencv_cannot_parse_fails_with_a_clear_error(value):
    # OpenCV aborts the whole process (std::stoull) on a negative or overflowing limit; refuse it before cv2 sees it.
    env = {**os.environ, "ARGUS_RETINA_MAX_PIXELS": value}
    code = "import argus_retina.sidecar"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 1 and "ARGUS_RETINA_MAX_PIXELS must be a positive pixel count" in out.stderr


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
    running, entered, overlap, results = [], [], [], []

    def load():
        def detect_all(images):
            running.append(1)
            entered.append(images)
            overlap.append(len(running))  # read in the test thread: a worker's assert would not fail the test
            started.set()
            release.wait(5)
            running.pop()
            return [[] for _ in images]

        return detect_all

    _, r = client(load)
    r.load()

    def detect(images):
        results.append(r.detect(images))

    t = threading.Thread(target=detect, args=([b"x"],))
    t.start()
    assert started.wait(5)
    h = r.health()
    assert h["busy"] is True and h["idle_s"] is None
    t2 = threading.Thread(target=detect, args=([b"y"],))
    t2.start()
    deadline = time.monotonic() + 5
    while r._busy < 2 and time.monotonic() < deadline:  # t2 was accepted and waits for the first call
        time.sleep(0.001)
    time.sleep(0.05)
    assert r._busy == 2 and t2.is_alive() and entered == [[b"x"]]  # waiting, not detecting
    release.set()
    t.join(5)
    t2.join(5)
    assert not t.is_alive() and not t2.is_alive()
    assert entered == [[b"x"], [b"y"]] and max(overlap) == 1 and running == [] and len(results) == 2
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


# --- GPU residents: the admin routes ---------------------------------------------------------------------------------


def gpu_client(loader=None, *, released=None):
    r = Resident(
        "Guard",
        loader or (lambda: lambda images: [[] for _ in images]),
        {"model": "m"},
        device="gpu",
        release=(lambda: released.append(1)) if released is not None else None,
        memory=lambda: {"free_vram_mb": 1000.0, "total_mb": 8000.0},
    )
    return TestClient(create_app(r, load=False)), r


def test_a_cpu_image_has_no_admin_routes():
    c, r = client(fake_loader([]))
    r.load()
    assert c.post("/admin/unload").status_code == 404


def test_a_gpu_resident_reports_gpu_and_its_memory():
    c, r = gpu_client()
    r.load()
    h = ResidentHealth.model_validate(c.get("/health").json())
    assert h.model_state == "gpu" and h.on_gpu and h.gpu is not None and h.gpu.total_mb == 8000
    assert c.get("/health").json()["device"] == "gpu"


def test_unload_frees_the_model_and_wake_loads_it_again():
    loads, released = [], []

    def load():
        loads.append(1)
        return lambda images: [[] for _ in images]

    c, r = gpu_client(load, released=released)
    r.load()
    reply = ResidentActionReply.model_validate(c.post("/admin/unload").json())
    assert reply.model_state == "unloaded" and reply.unloaded is True and released == [1]
    again = ResidentActionReply.model_validate(c.post("/admin/evict").json())  # evict is an unload here
    assert again.unloaded is False and released == [1]
    woke = ResidentActionReply.model_validate(c.post("/admin/wake").json())
    assert woke.model_state == "gpu" and woke.unloaded is None and loads == [1, 1]
    assert c.post("/admin/wake").status_code == 200 and loads == [1, 1]  # already loaded: nothing to do


def test_an_unloaded_gpu_model_loads_on_the_next_request():
    loads = []

    def load():
        loads.append(1)
        return lambda images: [[{"class": "x", "score": 1.0, "box": [0, 0, 1, 1]}] for _ in images]

    c, r = gpu_client(load)
    r.load()
    r.unload()
    assert c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))]).status_code == 200
    assert loads == [1, 1] and r.state == "gpu"


def test_unload_while_busy_is_a_409_and_does_nothing():
    started, release = threading.Event(), threading.Event()

    def load():
        def detect_all(images):
            started.set()
            release.wait(5)
            return [[] for _ in images]

        return detect_all

    c, r = gpu_client(load)
    r.load()
    t = threading.Thread(target=r.detect, args=([b"x"],))
    t.start()
    assert started.wait(5)
    try:
        resp = c.post("/admin/unload")
        assert resp.status_code == 409
        busy = ResidentBusyReply.model_validate(resp.json())
        assert busy.active == 1 and r.state == "gpu"
    finally:
        release.set()
        t.join(5)
    assert c.post("/admin/unload").status_code == 200


def test_admin_routes_while_loading_or_failed_are_503():
    gate = threading.Event()

    def slow_load():
        gate.wait(5)
        raise RuntimeError("CUDA out of memory")

    c, r = gpu_client(slow_load)
    t = r.start()
    try:
        while r.state != "loading":
            pass
        assert c.post("/admin/unload").status_code == 503
        assert c.post("/admin/wake").status_code == 503
    finally:
        gate.set()
        t.join(5)
    assert r.state == "error"
    resp = c.post("/admin/wake")  # tries again, and says why it failed
    assert resp.status_code == 503 and "out of memory" in resp.json()["detail"]


def test_admin_routes_need_the_token_when_one_is_set(monkeypatch):
    c, r = gpu_client()
    r.load()
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "s3cret")
    assert c.post("/admin/unload").status_code == 401
    assert c.post("/admin/unload", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/admin/unload", headers={"Authorization": "Bearer s3cret"}).status_code == 200


@pytest.mark.parametrize(
    ("body", "healthy"),
    [
        ({"model_state": "gpu"}, True),
        ({"model_state": "cpu"}, True),
        ({"model_state": "unloaded", "device": "gpu"}, True),  # argus-iris freed it; the next request loads it
        ({"model_state": "unloaded", "device": "cpu"}, False),
        ({"model_state": "loading"}, False),
        ({"model_state": "error"}, False),
        ([], False),
    ],
)
def test_the_healthcheck(body, healthy, monkeypatch):
    import io
    import json

    from argus_retina import healthcheck

    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(body).encode()))
    assert healthcheck.main() == (0 if healthy else 1)


# --- GPU leases --------------------------------------------------------------------------------------------------------


class Leases:
    """Records each lease and what was running under it."""

    def __init__(self, refuse: Exception | None = None):
        self.held: list[str] = []
        self.active = 0
        self.refuse = refuse

    def __call__(self):
        from contextlib import contextmanager

        @contextmanager
        def lease():
            if self.refuse is not None:
                raise self.refuse
            self.active += 1
            assert self.active == 1  # never two at once: the lease is taken inside the resident's lock
            try:
                yield
            finally:
                self.active -= 1

        return lease()


def leased_client(leases, loads):
    """A GPU resident under *leases*. *loads* and ``leases.held`` record whether a lease was held at each load and
    each detect."""

    def detect_all(images):
        leases.held.append(leases.active)
        return [[] for _ in images]

    def load():
        loads.append(leases.active)
        return detect_all

    r = Resident("Guard", load, {"model": "m"}, device="gpu", lease=leases)
    return TestClient(create_app(r, load=False)), r


def test_the_startup_load_and_each_request_hold_a_lease_and_wake_does_not():
    leases, loads = Leases(), []
    c, r = leased_client(leases, loads)
    r.load_leased()
    assert loads == [1] and r.state == "gpu"
    files = [("f1", ("a.png", png(), "image/png"))]
    assert c.post("/infer", files=files).status_code == 200 and leases.held == [1]
    r.unload()
    assert c.post("/infer", files=files).status_code == 200  # reloads under the request's lease
    assert loads == [1, 1] and leases.held == [1, 1]
    r.unload()
    assert c.post("/admin/wake").status_code == 200 and loads == [1, 1, 0]  # argus-iris made room: no lease


def test_concurrent_requests_take_one_lease_at_a_time():
    leases, loads = Leases(), []
    _, r = leased_client(leases, loads)
    r.load_leased()
    threads = [threading.Thread(target=r.detect, args=([b"x"],)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert leases.held == [1, 1, 1, 1]  # Leases asserts none overlapped


def test_a_refused_lease_is_a_503_and_leaves_the_model_unloaded():
    leases, loads = Leases(refuse=TimeoutError("broker queue full")), []
    c, r = leased_client(leases, loads)
    r.load_leased()
    assert r.state == "unloaded" and "broker queue full" in r.error and loads == [] and r.ready
    leases.refuse = None
    assert c.post("/admin/wake").status_code == 200  # iris's wake still loads it
    leases.refuse = TimeoutError("broker queue full")
    resp = c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))])
    assert resp.status_code == 503 and "no GPU lease" in resp.json()["detail"]
    assert r.health()["busy"] is False


def test_a_refused_startup_lease_loads_on_the_next_request():
    leases, loads = Leases(refuse=TimeoutError("broker queue full")), []
    c, r = leased_client(leases, loads)
    r.load_leased()
    leases.refuse = None  # argus-iris made room
    assert c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))]).status_code == 200
    assert loads == [1] and r.state == "gpu" and r.error == ""


def test_a_gpu_load_that_failed_on_demand_tries_again_on_the_next_request():
    calls = []

    def load():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return lambda images: [[] for _ in images]

    r = Resident("Guard", load, {}, device="gpu")
    r.load()
    assert r.state == "error"
    assert r.detect([b"x"]) == [[]] and r.state == "gpu" and len(calls) == 2


def test_an_error_inside_a_lease_is_not_a_lease_error():
    def load():
        def detect_all(images):
            raise RuntimeError("model blew up")

        return detect_all

    r = Resident("Guard", load, {}, device="gpu", lease=Leases())
    r.load()
    with pytest.raises(RuntimeError, match="model blew up"):
        r.detect([b"x"])


def test_infer_on_a_gpu_resident_that_failed_to_load_tries_again():
    calls = []

    def load():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return lambda images: [[] for _ in images]

    r = Resident("Guard", load, {}, device="gpu")
    c = TestClient(create_app(r, load=False))
    r.load()
    assert r.state == "error"
    assert c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))]).status_code == 200
    assert r.state == "gpu" and len(calls) == 2


def test_infer_on_a_cpu_resident_that_failed_to_load_stays_a_503():
    r = Resident("NudeNet", lambda: (_ for _ in ()).throw(RuntimeError("weights missing")), {})
    c = TestClient(create_app(r, load=False))
    r.load()
    assert c.post("/infer", files=[("f1", ("a.png", png(), "image/png"))]).status_code == 503
