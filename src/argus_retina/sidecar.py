"""The HTTP contract every retina image serves, around one detector.

``POST /infer`` takes multipart image fields (``f1``, ``f2``, ...) and answers NudeNet's shape,
``{"prediction": [[{"class", "score", "box": [x, y, w, h]}, ...], ...], "success": true}``, one list per image
in field order. ``argus_cortex.moderation`` parses it.

``GET /health`` answers argus-cortex's ``ResidentHealth`` (``model_state``, ``busy``, ``idle_s``), plus what the
image runs (``model`` and whatever else it names, e.g. ``resolution``, which ``NudeNetClient.info()`` reads), and
``version`` and ``commit``: AGPL section 13, the source this image was built from.

The model loads on a background thread at startup, so ``/health`` answers ``loading`` (then ``cpu``, or ``error``)
instead of the port staying closed. ``/infer`` answers 503 until it is loaded. One inference runs at a time.

Moved from argus-cull's ``sidecar-common/sidecar.py`` (argus-cortex#88).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

import argus_retina

__all__ = ["Detect", "Loader", "Resident", "create_app", "decode_bgr", "form_images"]

Detect = Callable[[list[bytes]], list[list[dict[str, Any]]]]
Loader = Callable[[], Detect]
State = Literal["unloaded", "loading", "cpu", "gpu", "error"]

# The source commit, for /health (AGPL section 13). The image's Dockerfile sets it from the build.
COMMIT_ENV = "ARGUS_RETINA_COMMIT"


async def form_images(form: Any) -> list[bytes]:
    """Multipart file parts in the order they were sent (``f1``, ``f2``, ...)."""
    blobs: list[bytes] = []
    for _key, item in form.multi_items():
        if hasattr(item, "read"):
            blobs.append(await item.read())
    return blobs


def decode_bgr(data: bytes) -> Any:
    """Three-channel BGR in stored pixel order (no EXIF rotation).

    Every detector decodes the same way, so a caller can compare one detector's boxes with another's. Grayscale
    frames come back as three channels.
    """
    import cv2  # noqa: PLC0415 - in the server extra
    import numpy as np  # noqa: PLC0415

    mat = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if mat is None:
        raise ValueError("unreadable image")
    return mat


class Resident:
    """One detector: loads it, runs one inference at a time, and reports its state for ``/health``."""

    def __init__(self, name: str, loader: Loader, info: dict[str, Any]):
        self.name = name
        self.info = info
        self._loader = loader
        self._detect: Detect | None = None
        self.state: State = "unloaded"
        self.error = ""
        self._lock = threading.Lock()  # one inference at a time
        self._busy = 0
        self._busy_lock = threading.Lock()
        self._last_done: float | None = None
        self.log = logging.getLogger(f"argus_retina.{name.lower()}")

    def load(self) -> None:
        self.state = "loading"
        try:
            self._detect = self._loader()
        except Exception as exc:  # reported on /health; the server stays up to say so
            self.log.exception("%s failed to load", self.name)
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"[:500]
            return
        self.state = "cpu"  # retina images run on the CPU (CUDA_VISIBLE_DEVICES is empty)
        self._last_done = time.monotonic()
        self.log.info("%s loaded: %s", self.name, self.info)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.load, name=f"load-{self.name}", daemon=True)
        t.start()
        return t

    def detect(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        if self._detect is None:
            raise RuntimeError(f"{self.name} is not loaded")
        with self._busy_lock:
            self._busy += 1
        try:
            with self._lock:
                return self._detect(images)
        finally:
            with self._busy_lock:
                self._busy -= 1
                self._last_done = time.monotonic()

    def health(self) -> dict[str, Any]:
        busy = self._busy > 0
        idle = None
        if not busy and self.state in ("cpu", "gpu") and self._last_done is not None:
            idle = round(time.monotonic() - self._last_done, 1)
        body: dict[str, Any] = {
            "resident_version": "1.0",
            "model_state": self.state,
            "busy": busy,
            "idle_s": idle,
            "detector": self.name,
            **self.info,
            "version": argus_retina.__version__,
            "commit": os.environ.get(COMMIT_ENV) or None,
        }
        if self.error:
            body["error"] = self.error
        return body


def create_app(resident: Resident, *, load: bool = True) -> FastAPI:
    """The FastAPI app around *resident*. ``load=False`` leaves loading to the caller (tests)."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        if load:
            resident.start()
        yield

    app = FastAPI(title=f"argus-retina {resident.name}", version=argus_retina.__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return resident.health()

    @app.post("/infer")
    async def infer(request: Request) -> dict[str, Any]:
        if resident.state != "cpu" and resident.state != "gpu":
            raise HTTPException(status_code=503, detail=f"{resident.name} is {resident.state}")
        images = await form_images(await request.form())
        if not images:
            raise HTTPException(status_code=400, detail="expected multipart image fields (f1, ...)")
        try:
            prediction = await run_in_threadpool(resident.detect, images)
        except ValueError as exc:  # decode_bgr: not an image
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            resident.log.exception("%s detect failed", resident.name)
            raise HTTPException(status_code=500, detail=f"{resident.name} detect failed") from exc
        return {"prediction": prediction, "success": True}

    return app


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
