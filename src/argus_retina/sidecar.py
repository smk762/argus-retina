"""The HTTP contract every retina image serves, around one detector.

``POST /infer`` takes multipart image fields (``f1``, ``f2``, ...) and answers NudeNet's shape,
``{"prediction": [[{"class", "score", "box": [x, y, w, h]}, ...], ...], "success": true}``, one list per image
in field order. ``argus_cortex.moderation`` parses it.

``GET /health`` answers argus-cortex's ``ResidentHealth`` (``model_state``, ``busy``, ``idle_s``), plus what the
image runs (``model`` and whatever else it names, e.g. ``resolution``, which ``NudeNetClient.info()`` reads), and
``version`` and ``commit``: AGPL section 13, the source this image was built from.

The model loads on a background thread at startup, so ``/health`` answers ``loading`` (then ``cpu`` or ``gpu``, or
``error``) instead of the port staying closed. ``/infer`` answers 503 until it is loaded. One inference runs at a time.

An image that holds VRAM (``Resident(..., device="gpu")``) also serves argus-cortex's resident admin routes,
``POST /admin/unload``, ``/admin/evict`` and ``/admin/wake``, behind ``Authorization: Bearer $ARGUS_ADMIN_TOKEN``
(open while the token is unset, as the suite's other residents are). They answer ``ResidentActionReply``, or 409
``ResidentBusyReply`` while work runs. After an unload the next request loads the model again.

Moved from argus-cull's ``sidecar-common/sidecar.py`` (argus-cortex#88).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import argus_retina

__all__ = [
    "Busy",
    "Detect",
    "Loader",
    "NotReady",
    "Resident",
    "UnreadableImage",
    "create_app",
    "decode_bgr",
    "form_images",
]

Detect = Callable[[list[bytes]], list[list[dict[str, Any]]]]
# What a loader returns: the detector /infer calls (Detect), or the model object a module's own route uses.
Loader = Callable[[], Any]
State = Literal["unloaded", "loading", "cpu", "gpu", "error"]
Device = Literal["cpu", "gpu"]
# A GPU lease from argus-cortex's coordinator (argus-iris's broker, a host lock, or nothing), held for a block of work.
Lease = Callable[[], AbstractContextManager[Any]]

# The source commit, for /health (AGPL section 13). The image's Dockerfile sets it from the build.
COMMIT_ENV = "ARGUS_RETINA_COMMIT"

# The largest image /infer decodes, in pixels: 2**27 (134 MP, about 400 MB as BGR) still admits a 108 MP phone photo.
# OpenCV's own default is 2**30 (about 3 GB as BGR), so a small, highly compressed upload declaring huge dimensions
# could OOM the server. OpenCV checks the header against OPENCV_IO_MAX_IMAGE_PIXELS before it allocates, but reads
# it when cv2 is imported, so it is set here: every image module imports this one before anything imports cv2.
# decode_bgr checks the decoded size too, so the budget holds (if not the allocation) should cv2 come first.
MAX_PIXELS_ENV = "ARGUS_RETINA_MAX_PIXELS"
MAX_PIXELS = int(os.environ.get(MAX_PIXELS_ENV, "") or 2**27)
if not 0 < MAX_PIXELS < 2**64:  # OpenCV parses it as a size_t and aborts the process at import on anything else
    raise ValueError(f"{MAX_PIXELS_ENV} must be a positive pixel count, not {MAX_PIXELS}")
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(MAX_PIXELS)


class UnreadableImage(ValueError):
    """A part that doesn't decode as an image: the caller's fault (422), unlike a ValueError inside a model."""


class _Entered:
    """A context manager whose ``__enter__`` failing (the lease refused or timed out) is ``NotReady``, while an error in
    the body passes through as itself."""

    def __init__(self, cm: AbstractContextManager[Any]):
        self._cm = cm

    def __enter__(self) -> Any:
        try:
            return self._cm.__enter__()
        except Exception as exc:
            raise NotReady(f"no GPU lease: {exc}") from exc

    def __exit__(self, *exc_info: Any) -> Any:
        return self._cm.__exit__(*exc_info)


class Busy(RuntimeError):
    """An admin action refused because work is running: nothing was done (409)."""

    def __init__(self, active: int):
        super().__init__(f"{active} running")
        self.active = active


class NotReady(RuntimeError):
    """The model is loading or failed to load (503)."""


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
    frames come back as three channels. Raises ``UnreadableImage`` for an empty part, one OpenCV can't decode, and
    one larger than ``MAX_PIXELS``.
    """
    import cv2  # noqa: PLC0415 - in the server extra
    import numpy as np  # noqa: PLC0415

    if not data:
        raise UnreadableImage("empty image")
    try:
        # cv2.error: a codec failing on corrupt data, or a header over OPENCV_IO_MAX_IMAGE_PIXELS
        mat = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    except cv2.error as exc:
        if (
            getattr(exc, "code", None) == cv2.Error.StsNoMem
        ):  # the server ran out of memory: a 500, not the caller's fault
            raise
        raise UnreadableImage(f"unreadable image (or over {MAX_PIXELS} pixels)") from exc
    if mat is None:
        raise UnreadableImage("unreadable image")
    if mat.shape[0] * mat.shape[1] > MAX_PIXELS:
        raise UnreadableImage(f"image over {MAX_PIXELS} pixels")
    return mat


class Resident:
    """One model: loads it, runs one inference at a time, and reports its state for ``/health``.

    *device* is where the loaded model lives. ``"cpu"`` images never unload; a ``"gpu"`` one can be unloaded (and
    woken) by argus-iris through the admin routes, and loads again on the next request. *release* runs after an
    unload, once the model is dropped (``gc`` and ``torch.cuda.empty_cache``); *memory* returns ``/health``'s ``gpu``
    block (cortex's ``ResidentGpu``), or None when it can't be read.

    *lease*, on a GPU image, holds a GPU lease (``argus_cortex.gpu``) around the startup load and around each request,
    including the load a request after an unload does, so argus-iris can make room first. It is taken inside the
    one-at-a-time lock: one request, one lease. ``wake`` takes none: argus-iris calls it, having made room itself. A
    lease that can't be had is ``NotReady`` (503), and the reason is kept.
    """

    def __init__(
        self,
        name: str,
        loader: Loader,
        info: dict[str, Any],
        *,
        device: Device = "cpu",
        release: Callable[[], None] | None = None,
        memory: Callable[[], dict[str, float] | None] | None = None,
        lease: Lease | None = None,
    ):
        self.name = name
        self.info = info
        self.device: Device = device
        self._loader = loader
        self._release = release
        self._memory = memory
        self._lease = lease
        self._model: Any = None
        self.state: State = "unloaded"
        self.error = ""
        self._lock = threading.Lock()  # one inference, or one unload, at a time
        self._load_lock = threading.Lock()  # one load at a time
        self._busy = 0
        self._busy_lock = threading.Lock()
        self._last_done: float | None = None
        self.log = logging.getLogger(f"argus_retina.{name.lower()}")

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            self.state, self.error = "loading", ""
            try:
                model = self._loader()
            except Exception as exc:  # reported on /health; the server stays up to say so
                self.log.exception("%s failed to load", self.name)
                self.state, self.error = "error", f"{type(exc).__name__}: {exc}"[:500]
                return
            self._model = model
            # CPU images keep CUDA_VISIBLE_DEVICES empty; a GPU image's loader puts the model on the GPU or fails.
            self.state = self.device
            self._last_done = time.monotonic()
            self.log.info("%s loaded: %s", self.name, self.info)

    def _leased(self) -> AbstractContextManager[Any]:
        if self._lease is None:
            return nullcontext()
        try:
            return _Entered(self._lease())
        except Exception as exc:
            raise NotReady(f"no GPU lease for {self.name}: {exc}") from exc

    def load_leased(self) -> None:
        """``load`` under a GPU lease: the startup load. A lease that can't be had is reported like a failed load."""
        try:
            with self._leased():
                self.load()
        except NotReady as exc:
            self.log.warning("%s", exc)
            self.state, self.error = "error", str(exc)[:500]

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.load_leased, name=f"load-{self.name}", daemon=True)
        t.start()
        return t

    def run(self, work: Callable[[Any], Any]) -> Any:
        """``work(model)``, one call at a time. A GPU model unloaded by an admin loads again first.

        Raises ``NotReady`` while the model is loading or failed to load.
        """
        with self._busy_lock:
            self._busy += 1
        try:
            with self._lock, self._leased():
                if self._model is None and self.device == "gpu" and self.state == "unloaded":
                    self.load()
                model = self._model
                if model is None:
                    raise NotReady(f"{self.name} is {self.state}")
                return work(model)
        finally:
            with self._busy_lock:
                self._busy -= 1
                self._last_done = time.monotonic()

    def detect(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        return self.run(lambda detect: detect(images))

    @property
    def ready(self) -> bool:
        """A request would be served: the model is loaded, or (on a GPU image) unloaded and loads on demand."""
        return self.state in ("cpu", "gpu") or (self.device == "gpu" and self.state == "unloaded")

    def unload(self) -> bool:
        """Drop the model and free what it held. True when something was freed, False when it was already unloaded.

        Raises ``Busy`` while work runs or waits, and ``NotReady`` while the model is loading.
        """
        if not self._load_lock.acquire(blocking=False):
            raise NotReady(f"{self.name} is loading")
        try:
            if not self._lock.acquire(blocking=False):
                raise Busy(max(self._busy, 1))
            try:
                if self._busy:  # a request waiting for the lock counts: it was accepted
                    raise Busy(self._busy)
                freed, self._model = self._model is not None, None
                if freed and self._release is not None:
                    self._release()
                self.state, self.error = "unloaded", ""
            finally:
                self._lock.release()
        finally:
            self._load_lock.release()
        self.log.info("%s unloaded (freed: %s)", self.name, freed)
        return freed

    def wake(self) -> None:
        """Load the model now, if it isn't loaded. Raises ``Busy``, or ``NotReady`` when it is loading or fails."""
        if self._model is not None:
            return
        if self.state == "loading":
            raise NotReady(f"{self.name} is loading")
        if not self._lock.acquire(blocking=False):
            raise Busy(max(self._busy, 1))
        try:
            self.load()
        finally:
            self._lock.release()
        if self._model is None:
            raise NotReady(
                f"{self.name} is {self.state}: {self.error}" if self.error else f"{self.name} is {self.state}"
            )

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
            "device": self.device,
            **self.info,
            "version": argus_retina.__version__,
            "commit": os.environ.get(COMMIT_ENV) or None,
        }
        if self._memory is not None:
            try:
                gpu = self._memory()
            except Exception:  # memory figures are a nicety; /health must still answer
                self.log.debug("reading GPU memory failed", exc_info=True)
                gpu = None
            if gpu:
                body["gpu"] = gpu
        if self.error:
            body["error"] = self.error
        return body

    def action_reply(self, unloaded: bool | None) -> dict[str, Any]:
        """``ResidentActionReply``: the state after an admin action."""
        return {"resident_version": "1.0", "status": "ok", "model_state": self.state, "unloaded": unloaded}


def create_app(resident: Resident, *, load: bool = True, infer: bool = True) -> FastAPI:
    """The FastAPI app around *resident*. ``load=False`` leaves loading to the caller (tests).

    ``infer=False`` leaves out ``POST /infer``, for an image that serves its own route (SingGuard). A GPU resident
    also gets the admin routes.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        if load:
            resident.start()
        yield

    app = FastAPI(title=f"argus-retina {resident.name}", version=argus_retina.__version__, lifespan=lifespan)

    # async: answered on the event loop, so /infer calls queued on the threadpool can't starve the HEALTHCHECK.
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return resident.health()

    if infer:

        @app.post("/infer")
        async def infer_route(request: Request) -> dict[str, Any]:
            if not resident.ready:
                raise HTTPException(status_code=503, detail=f"{resident.name} is {resident.state}")
            async with request.form() as form:  # closes the spooled upload files
                images = await form_images(form)
            if not images:
                raise HTTPException(status_code=400, detail="expected multipart image fields (f1, ...)")
            try:
                prediction = await run_in_threadpool(resident.detect, images)
            except UnreadableImage as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except NotReady as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                resident.log.exception("%s detect failed", resident.name)
                raise HTTPException(status_code=500, detail=f"{resident.name} detect failed") from exc
            return {"prediction": prediction, "success": True}

    if resident.device == "gpu":
        add_admin_routes(app, resident)

    return app


def add_admin_routes(app: FastAPI, resident: Resident) -> None:
    """``/admin/unload``, ``/admin/evict`` and ``/admin/wake`` (argus-cortex's resident contract).

    Evict is an unload here: a GPU model on this server has no CPU stage to keep it warm in (a 4-bit bitsandbytes
    model can't move to the CPU), so a manifest declares only ``unload_path`` and ``warm_path``.
    """
    from argus_cortex.auth import bearer  # noqa: PLC0415 - GPU images install argus-cortex; CPU images need no admin

    admin = [Depends(bearer("ARGUS_ADMIN_TOKEN", open_when_unset=True))]

    def act(action: Callable[[], bool | None]) -> Callable[[], Any]:
        async def route() -> Any:
            try:
                unloaded = await run_in_threadpool(action)
            except Busy as exc:
                body = {"resident_version": "1.0", "status": "busy", "active": exc.active}
                return JSONResponse(status_code=409, content=body)
            except NotReady as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return resident.action_reply(unloaded)

        return route

    app.add_api_route("/admin/unload", act(resident.unload), methods=["POST"], dependencies=admin)
    app.add_api_route("/admin/evict", act(resident.unload), methods=["POST"], dependencies=admin)
    app.add_api_route("/admin/wake", act(resident.wake), methods=["POST"], dependencies=admin)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
