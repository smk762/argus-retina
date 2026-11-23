"""Community Forensics (ViT-S/16 at 384, CPU): ``uvicorn argus_retina.synthetic:app``.

``POST /synthetic`` scores each image with the probability the model assigns to "generated". The caller chooses any
cutoff. This server returns no label.

The weights are Community Forensics (`OwensLab/commfor-model-384`, MIT), a ViT-S trained on 2.7 million images from
4,803 generators (Park and Owens, CVPR 2025). They are baked in, pinned by Hub commit and sha256. The head is one
logit; ``sigmoid`` of it is the probability. Preprocessing is the paper's eval transform: shortest side to 440,
centre crop 384, ImageNet mean and standard deviation. A rehost that uses CLIP's mean instead does not match these
weights.

Settings: ``SYNTHETIC_MODEL`` (default ``/models/synthetic/model.safetensors``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from argus_retina.sidecar import (
    NotReady,
    Resident,
    UnreadableImage,
    configure_logging,
    create_app,
    decode_bgr,
    form_images,
)

SCHEMA = "argus-retina.synthetic/v1"
# OwensLab/commfor-model-384. The sha256 is the safetensors file at this commit (verified against the blob).
REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
# timm's name for the architecture the checkpoint was trained from. pretrained=False: the baked file replaces it.
ARCH = "vit_small_patch16_384.augreg_in21k_ft_in1k"
# torchvision Resize(440) then CenterCrop(384), then ImageNet normalisation. PIL bilinear, as torchvision uses on a
# PIL image.
RESIZE = 440
CROP = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL = Path(os.environ.get("SYNTHETIC_MODEL", "/models/synthetic/model.safetensors"))


def resized_size(width: int, height: int, short: int = RESIZE) -> tuple[int, int]:
    """``torchvision.transforms.Resize(short)``: the shorter side becomes *short*, the other is truncated, not rounded."""
    if width <= height:
        return short, int(short * height / width)
    return int(short * width / height), short


def crop_origin(width: int, height: int, crop: int = CROP) -> tuple[int, int]:
    """``CenterCrop``'s top-left, with its rounded offset (Python's half-to-even ``round``)."""
    return int(round((width - crop) / 2.0)), int(round((height - crop) / 2.0))


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8, as ``decode_bgr`` returns it, to a ``(3, 384, 384)`` float32 tensor in the model's normalisation."""
    image = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
    image = image.resize(resized_size(*image.size), Image.Resampling.BILINEAR)
    left, top = crop_origin(*image.size)
    image = image.crop((left, top, left + CROP, top + CROP))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))


def to_scores(probs: list[float]) -> list[dict[str, float]]:
    """One ``{"synthetic": p}`` per probability, four decimal places, as the other detectors report a score."""
    return [{"synthetic": round(float(p), 4)} for p in probs]


def detector_state(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``vit.`` prefix the published ``ViTClassifier`` saves around the timm module."""
    if raw and all(isinstance(key, str) and key.startswith("vit.") for key in raw):
        return {key.removeprefix("vit."): value for key, value in raw.items()}
    return raw


def score(model: Any, images: list[bytes]) -> list[dict[str, float]]:
    """Each image's generated-probability. An unreadable part raises before the model runs."""
    frames = [preprocess(decode_bgr(data)) for data in images]
    import torch  # noqa: PLC0415 - only in the synthetic image

    batch = torch.from_numpy(np.stack(frames))
    with torch.inference_mode():
        probs = torch.sigmoid(model(batch).reshape(-1))
    return to_scores([float(p) for p in probs])


def load() -> Any:
    if not MODEL.is_file():
        raise FileNotFoundError(f"synthetic model not found: {MODEL}")
    import timm  # noqa: PLC0415 - only in the synthetic image
    from safetensors.torch import load_file  # noqa: PLC0415 - only in the synthetic image

    net = timm.create_model(ARCH, pretrained=False, num_classes=1)
    net.load_state_dict(detector_state(load_file(str(MODEL))))
    net.eval()
    return net


resident = Resident(
    "Synthetic",
    load,
    {"model": "commfor-model-384", "revision": REVISION, "input": CROP, "schema": SCHEMA},
)
configure_logging()
app = create_app(resident, infer=False)


async def synthetic_route(request: Request) -> dict[str, Any]:
    if not resident.accepting:
        raise HTTPException(status_code=503, detail=f"{resident.name} is {resident.state}")
    async with request.form() as form:
        images = await form_images(form)
    if not images:
        raise HTTPException(status_code=400, detail="expected multipart image fields (f1, ...)")
    try:
        scores = await run_in_threadpool(resident.run, lambda model: score(model, images))
    except UnreadableImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        resident.log.exception("synthetic score failed")
        raise HTTPException(status_code=500, detail="Synthetic score failed") from exc
    return {"schema": SCHEMA, "scores": scores, "success": True}


app.add_api_route("/synthetic", synthetic_route, methods=["POST"])
