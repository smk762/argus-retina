"""What each image declares: its tenant manifest, the YOLO detections, and the Dockerfiles' pins."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from argus_cortex.gpu_wire import load_tenant_manifest

from argus_retina.yolo import boxes

ROOT = Path(__file__).parent.parent
IMAGES = ["nudenet", "erax", "watermark"]
NUDENET_640M_SHA256 = "04fe3d77980780c1f8297dc6d7f942fd5b3abe6942a188f742a85241e4f634eb"


@pytest.mark.parametrize("image", IMAGES)
def test_each_image_has_a_valid_cpu_only_tenant_manifest(image):
    m = load_tenant_manifest(ROOT / "images" / image / "gpu-tenant.yaml")
    assert m.tenant == f"argus-retina-{image}" and m.repo == "smk762/argus-retina"
    assert m.residents == () and all(k.vram_mb == 0 for k in m.kinds)  # CPU images hold no VRAM


@pytest.mark.parametrize("image", IMAGES)
def test_each_dockerfile_runs_its_module_on_the_cpu(image):
    text = (ROOT / "images" / image / "Dockerfile").read_text()
    assert f"argus_retina.{image}:app" in text
    assert "CUDA_VISIBLE_DEVICES= " in text or "CUDA_VISIBLE_DEVICES= \\" in text
    assert "ARGUS_RETINA_COMMIT=${COMMIT}" in text  # /health names the source (AGPL section 13)
    assert "USER retina" in text


def test_nudenet_weights_are_pinned_the_same_in_the_dockerfile_and_makefile():
    docker = (ROOT / "images" / "nudenet" / "Dockerfile").read_text()
    make = (ROOT / "Makefile").read_text()
    assert f"NUDENET_640M_SHA256={NUDENET_640M_SHA256}" in docker
    assert f"NUDENET_640M_SHA256 := {NUDENET_640M_SHA256}" in make


def test_erax_weights_are_pinned_to_a_hub_commit_and_checksum():
    text = (ROOT / "images" / "erax" / "Dockerfile").read_text()
    assert re.search(r"EraX-NSFW-V1\.0/resolve/[0-9a-f]{40}/erax_nsfw_yolo11s\.pt", text)
    assert re.search(r"ERAX_SHA256=[0-9a-f]{64}", text)


def test_the_watermark_image_bakes_in_no_weights():
    # fancyfeast's YOLO11x has no stated licence: it is mounted at runtime, never redistributed (argus-cortex#88)
    text = (ROOT / "images" / "watermark" / "Dockerfile").read_text()
    steps = [line for line in text.splitlines() if line.startswith(("RUN", "COPY", "ADD", "    &&"))]
    assert not any("curl" in s or "wget" in s or "weights" in s or ".pt" in s for s in steps)
    assert "WATERMARK_MODEL=/models/watermark.pt" in text
    assert "images/watermark/weights" in (ROOT / ".dockerignore").read_text()


def test_the_watermark_weights_for_a_local_mount_are_pinned_to_a_space_commit_and_checksum():
    make = (ROOT / "Makefile").read_text()
    assert re.search(r"joycaption-watermark-detection/resolve/[0-9a-f]{40}/yolo11x-train28-best\.pt", make)
    assert re.search(r"WATERMARK_SHA256 := [0-9a-f]{64}", make)


class _T:
    """A tensor stand-in: indexable, with tolist()."""

    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        return _T(self.v[i]) if isinstance(self.v[i], list) else self.v[i]

    def tolist(self):
        return self.v


def test_yolo_detections_are_nudenet_shaped():
    box = SimpleNamespace(xyxy=_T([[10.6, 20.2, 50.9, 80.0]]), cls=_T([2]), conf=_T([0.875]))
    results = [
        SimpleNamespace(names={2: "nipple"}, boxes=[box]),
        SimpleNamespace(names={}, boxes=None),
    ]
    assert boxes(results) == [{"class": "nipple", "score": 0.875, "box": [10, 20, 40, 59]}]
