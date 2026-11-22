"""InsightFace and MiVOLO: ``uvicorn argus_retina.faces:app``.

``POST /faces`` takes multipart image fields (``f1``, ``f2``, ...) and returns, per face, a box, the ArcFace
embedding, the detector keypoints and head pose, and MiVOLO's age and gender. Clustering, the under-18 rule and which
reading wins stay with the caller (curator). Thresholds do too: this only senses.

The weights are not in the image. InsightFace's ``buffalo_l`` pack has no licence that allows redistributing it, so
it is mounted at ``INSIGHTFACE_ROOT`` (``/models/insightface/models/buffalo_l``). MiVOLO v2 is Apache-2.0; its ONNX
is produced by ``argus-curator mivolo-convert`` and mounted at ``MIVOLO_MODEL``. Neither file is fetched.

Measured on a 3090 with the arena cap (curator, argus-curator#189): about 900 MB once loaded and 1,470 MB at peak.
The CUDA arena grows only by what an allocation asks, up to ``FACES_GPU_MEM_LIMIT_MB`` per session, and ONNX Runtime
does not hand it back on unload: argus-iris should stop the container to free the card.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from argus_retina.sidecar import (
    NotReady,
    Resident,
    UnreadableImage,
    configure_logging,
    create_app,
    decode_bgr,
    form_images,
)

SCHEMA = "argus-retina.faces/v1"
# What curator loads from the pack: boxes and keypoints, identity embeddings, and the 3D-68 landmarks behind pose.
# The pack's own gender/age model (MiVOLO replaces it) and its 2D-106 landmarks are not loaded.
MODULES = ("detection", "recognition", "landmark_3d_68")
GENDERS = ("male", "female")
# MiVOLO v2 config: age = raw * (max - min) + avg, then clipped.
_MIN_AGE, _MAX_AGE, _AVG_AGE = 0.0, 122.0, 61.0
INPUT_SIZE = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Lease ask: curator's measured peak (1,470 MB) plus headroom, and the weights that stay loaded.
VRAM_MB = 1600
MODEL_MB = 1000
# iBUG 68: each eye (outer corner first) and the inner lip.
_RIGHT_EYE = (36, 37, 38, 39, 40, 41)
_LEFT_EYE = (42, 43, 44, 45, 46, 47)

ROOT = Path(os.environ.get("INSIGHTFACE_ROOT", "/models/insightface"))
PACK = os.environ.get("FACES_PACK", "buffalo_l")
MIVOLO = Path(os.environ.get("MIVOLO_MODEL", "/models/faces/mivolo_v2_d1_384.onnx"))
DET_THRESH = float(os.environ.get("FACES_DET_THRESH", "0.5") or "0.5")
GPU_MEM_LIMIT_MB = int(os.environ.get("FACES_GPU_MEM_LIMIT_MB", "") or 1024)


def pack_dir() -> Path:
    """Where ``buffalo_l``'s ONNX files are mounted: ``<root>/models/<pack>``."""
    return ROOT / "models" / PACK


def cuda_options(limit_mb: int = GPU_MEM_LIMIT_MB) -> dict[str, str]:
    """ONNX Runtime CUDA arena: grow by the request, and stop at *limit_mb*.

    The default arena doubles without a ceiling and never returns memory. One curator process reached about 20 GB
    that way while each lease asked for 512 MB (argus-curator#189).
    """
    return {"arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(limit_mb * 1024 * 1024)}


def _providers() -> list[Any]:
    import onnxruntime as ort  # noqa: PLC0415 - in the faces image

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available")
    return [("CUDAExecutionProvider", cuda_options()), "CPUExecutionProvider"]


def xyxy_to_xywh(bbox: Any) -> list[float]:
    """InsightFace's ``[x1, y1, x2, y2]`` as the suite's ``[x, y, w, h]``."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def _pose(face: Any) -> dict[str, float | None]:
    """``buffalo_l`` stores ``pose`` as ``[pitch, yaw, roll]`` degrees. Older packs omit it."""
    raw = getattr(face, "pose", None)
    out: dict[str, float | None] = {"yaw": None, "pitch": None, "roll": None}
    if raw is None:
        return out
    try:
        out["pitch"] = round(float(raw[0]), 2)
        out["yaw"] = round(float(raw[1]), 2)
    except (TypeError, ValueError, IndexError):
        return {"yaw": None, "pitch": None, "roll": None}
    try:
        out["roll"] = round(float(raw[2]), 2)
    except (TypeError, ValueError, IndexError):
        out["roll"] = None
    return out


def _points(value: Any, width: int) -> list[list[float]] | None:
    try:
        pts = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if pts.ndim != 2 or pts.shape[1] < width or pts.shape[0] == 0:
        return None
    return [[round(float(n), 2) for n in row[:width]] for row in pts]


def _eye_aspect(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    width = float(np.linalg.norm(p1 - p4))
    if width <= 1e-6:
        return 0.0
    return (float(np.linalg.norm(p2 - p6)) + float(np.linalg.norm(p3 - p5))) / (2.0 * width)


def eye_mouth_open(landmarks: Any) -> tuple[float | None, float | None]:
    """``(eye_open, mouth_open)`` from ``landmark_3d_68``, or ``(None, None)``.

    ``eye_open`` is the eye aspect ratio averaged over both eyes. ``mouth_open`` is the mean inner-lip gap over the
    inner mouth width. Both are read in the image plane.
    """
    try:
        pts = np.asarray(landmarks, dtype=np.float64)[:, :2]
    except (TypeError, ValueError, IndexError):
        return None, None
    if pts.shape != (68, 2):
        return None, None
    eye = (_eye_aspect(pts, _RIGHT_EYE) + _eye_aspect(pts, _LEFT_EYE)) / 2.0
    width = float(np.linalg.norm(pts[60] - pts[64]))
    gap = sum(float(np.linalg.norm(pts[a] - pts[b])) for a, b in ((61, 67), (62, 66), (63, 65))) / 3.0
    mouth = gap / width if width > 1e-6 else 0.0
    return round(eye, 4), round(mouth, 4)


def face_record(face: Any) -> dict[str, Any]:
    """One InsightFace face, without MiVOLO. Boxes are ``[x, y, w, h]`` in the image's pixels."""
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    embedding = None if emb is None else [round(float(v), 5) for v in np.asarray(emb).reshape(-1)]
    landmarks = getattr(face, "landmark_3d_68", None)
    eye, mouth = eye_mouth_open(landmarks)
    return {
        "bbox": xyxy_to_xywh(face.bbox),
        "det_score": round(float(face.det_score), 4),
        "embedding": embedding,
        "kps": _points(getattr(face, "kps", None), 2),
        "pose": _pose(face),
        "landmark_3d_68": _points(landmarks, 3),
        "eye_open": eye,
        "mouth_open": mouth,
        "age_years": None,
        "gender": None,
        "gender_score": None,
    }


def decode_mivolo(output: Any) -> list[tuple[str, float, float]]:
    """``(gender, gender_score, age_years)`` per row. Gender order is MiVOLO's ``[male, female]``."""
    raw = np.asarray(output, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"MiVOLO output has shape {raw.shape}, expected (N, 3+)")
    shifted = raw[:, :2] - raw[:, :2].max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    years = np.clip(raw[:, 2] * (_MAX_AGE - _MIN_AGE) + _AVG_AGE, _MIN_AGE, _MAX_AGE)
    found = []
    for i in range(raw.shape[0]):
        choice = int(np.argmax(probs[i]))
        found.append((GENDERS[choice], round(float(probs[i, choice]), 4), round(float(years[i]), 2)))
    return found


def _letterbox(rgb: np.ndarray) -> np.ndarray:
    """Fit an RGB crop into a black 384 square, aspect preserved, centred. MiVOLO's ``class_letterbox``."""
    import cv2  # noqa: PLC0415 - in the server extra

    height, width = rgb.shape[:2]
    scale = min(INPUT_SIZE / height, INPUT_SIZE / width)
    nw, nh = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (width, height) else rgb
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    left = int(round((INPUT_SIZE - nw) / 2 - 0.1))
    top = int(round((INPUT_SIZE - nh) / 2 - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def _channels(rgb: np.ndarray) -> np.ndarray:
    pixels = _letterbox(rgb).astype(np.float32) / 255.0
    return ((pixels - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def _crop(bgr: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """The detector box clamped to the frame, as RGB. MiVOLO takes it unaligned."""
    x, y, w, h = bbox
    height, width = bgr.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(width, int(x + w)), min(height, int(y + h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return bgr[y1:y2, x1:x2, ::-1]


def mivolo_batch(bgr: np.ndarray, boxes: list[list[float]]) -> tuple[np.ndarray, list[int]]:
    """``(N, 6, 384, 384)`` face channels plus an absent person, and the indexes those rows came from.

    Face-only is a supported MiVOLO mode: the person channels are the black stand-in the checkpoint was given when
    there is no body. A box smaller than 2 px is left out.
    """
    crops = [_crop(bgr, box) for box in boxes]
    idx = [i for i, crop in enumerate(crops) if crop is not None]
    if not idx:
        return np.empty((0, 6, INPUT_SIZE, INPUT_SIZE), dtype=np.float32), idx
    faces = np.stack([_channels(crops[i]) for i in idx])
    absent = ((np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.float32) - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    persons = np.stack([absent for _ in idx])
    return np.concatenate([faces, persons], axis=1).astype(np.float32), idx


def apply_mivolo(records: list[dict[str, Any]], output: Any, idx: list[int]) -> None:
    """Write MiVOLO onto the records whose crops were readable."""
    if not idx:
        return
    decoded = decode_mivolo(output)
    if len(decoded) != len(idx):
        raise ValueError(f"MiVOLO returned {len(decoded)} rows for {len(idx)} faces")
    for row, i in zip(decoded, idx, strict=True):
        gender, score, years = row
        records[i]["gender"] = gender
        records[i]["gender_score"] = score
        records[i]["age_years"] = years


def detect(models: Any, images: list[bytes]) -> list[list[dict[str, Any]]]:
    """Faces for each image, in order. ``models`` is the ``(insightface app, mivolo session)`` pair ``load`` built."""
    frames = [decode_bgr(data) for data in images]
    app, session = models
    found: list[list[dict[str, Any]]] = []
    input_name = None
    for bgr in frames:
        records = [face_record(face) for face in app.get(bgr)]
        batch, idx = mivolo_batch(bgr, [rec["bbox"] for rec in records])
        if idx:
            if input_name is None:
                input_name = session.get_inputs()[0].name
            apply_mivolo(records, session.run(None, {input_name: batch})[0], idx)
        found.append(records)
    return found


def load() -> Any:
    pack = pack_dir()
    if not pack.is_dir() or not any(pack.glob("*.onnx")):
        raise FileNotFoundError(f"InsightFace pack not found: {pack} (mount {PACK}; this image does not download it)")
    if not MIVOLO.is_file():
        raise FileNotFoundError(f"MiVOLO model not found: {MIVOLO}")
    import onnxruntime as ort  # noqa: PLC0415 - in the faces image
    from insightface.app import FaceAnalysis  # noqa: PLC0415 - in the faces image

    providers = _providers()
    app = FaceAnalysis(name=PACK, root=str(ROOT), providers=providers, allowed_modules=list(MODULES))
    app.prepare(ctx_id=0, det_thresh=DET_THRESH, det_size=(640, 640))
    session = ort.InferenceSession(str(MIVOLO), providers=providers)
    return app, session


_coordinator: Any = None


def lease() -> Any:
    """A ``faces`` lease from the host's GPU coordinator, built once."""
    global _coordinator
    if _coordinator is None:
        from argus_cortex.gpu import coordinator_from_env  # noqa: PLC0415 - in the faces image

        _coordinator = coordinator_from_env()
    return _coordinator.lease(
        caller="argus-retina-faces",
        min_vram_mb=VRAM_MB,
        model_mb=MODEL_MB,
        kind="faces",
        priority="interactive",
    )


def release() -> None:
    """Drop the sessions. The CUDA arena stays until the process exits; iris stops the container to take it back."""
    gc.collect()


def lock_warning(env: Any = os.environ) -> str | None:
    """Why the GPU lease's lock-file fallback would fail here, or None. Same constraint as SingGuard (#20)."""
    coordinator = env.get("ARGUS_GPU_COORDINATOR", "").strip().lower()
    fallback = env.get("ARGUS_GPU_BROKER_FALLBACK", "lease").strip().lower() or "lease"
    if coordinator != "lease" and not (coordinator == "broker" and fallback == "lease"):
        return None
    path = env.get("ARGUS_GPU_LEASE_PATH", "").strip()
    if not path:
        from argus_cortex.gpu import default_lock_path  # noqa: PLC0415 - in the faces image

        path = default_lock_path()
    lock = Path(path)
    if lock.exists():
        if os.access(lock, os.R_OK | os.W_OK):
            return None
        problem = "can't be opened"
    elif os.access(lock.parent, os.W_OK | os.X_OK):
        return None
    else:
        problem = "can't be created (its directory isn't writable)"
    return (
        f"the GPU lease lock {lock} {problem} by uid {os.getuid()}, so a lease that falls back to it will be refused: "
        "make the host's ARGUS_GPU_LOCK_DIR mode 1777 (see the README)"
    )


resident = Resident(
    "Faces",
    load,
    {
        "model": PACK,
        "schema": SCHEMA,
        "det": 640,
        "det_thresh": DET_THRESH,
        "gpu_mem_limit_mb": GPU_MEM_LIMIT_MB,
    },
    device="gpu",
    release=release,
    lease=lease,
)
configure_logging()
if (_lock_problem := lock_warning()) is not None:
    resident.log.warning("%s", _lock_problem)
app = create_app(resident, infer=False)


async def faces_route(request: Request) -> dict[str, Any]:
    if not resident.accepting:
        raise HTTPException(status_code=503, detail=f"{resident.name} is {resident.state}")
    async with request.form() as form:
        images = await form_images(form)
    if not images:
        raise HTTPException(status_code=400, detail="expected multipart image fields (f1, ...)")
    try:
        found = await run_in_threadpool(resident.run, lambda models: detect(models, images))
    except UnreadableImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        resident.log.exception("faces detect failed")
        raise HTTPException(status_code=500, detail="Faces detect failed") from exc
    return {"schema": SCHEMA, "faces": found, "success": True}


app.add_api_route("/faces", faces_route, methods=["POST"])
