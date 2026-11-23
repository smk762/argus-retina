"""POST /c2pa: the signed claims, and no verdict."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from argus_retina.c2pa import SCHEMA, absent, failure_codes, to_report, unreadable
from argus_retina.sidecar import Resident

TRAINED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


def png() -> bytes:
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def manifest() -> dict:
    return {
        "claim_generator": "Adobe_Firefly 1.0",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {"action": "c2pa.created", "digitalSourceType": TRAINED},
                        {"action": "c2pa.edited", "parameters": {"digitalSourceType": TRAINED}},
                        {"action": "c2pa.color_adjustments"},
                    ]
                },
            },
            {"label": "c2pa.hash.data", "data": {"exclusions": []}},
        ],
        "ingredients": [{"title": "a.jpg"}, "nope"],
        "signature_info": {"issuer": "Adobe", "time": "2026-01-01T00:00:00Z", "alg": "Es256"},
    }


def test_a_manifest_reports_the_claims_and_not_the_rest_of_the_store():
    report = to_report(
        manifest(),
        "Invalid",
        {"activeManifest": {"failure": [{"code": "assertion.dataHash.mismatch", "explanation": "long"}]}},
    )
    assert report["present"] is True
    assert report["validation_state"] == "Invalid"
    assert report["failures"] == ["assertion.dataHash.mismatch"]
    assert report["claim_generator"] == "Adobe_Firefly 1.0"
    assert report["digital_source_types"] == [TRAINED]
    assert report["actions"] == ["c2pa.created", "c2pa.edited", "c2pa.color_adjustments"]
    assert report["ingredients"] == 2
    assert report["signature"] == {"issuer": "Adobe", "time": "2026-01-01T00:00:00Z"}
    assert report["error"] is None


def test_an_absent_report_and_a_library_error_share_one_shape():
    empty = absent()
    assert empty["present"] is False and empty["validation_state"] is None and empty["error"] is None
    broken = unreadable(RuntimeError("type is unsupported"))
    assert broken["present"] is False and broken["error"].startswith("RuntimeError:")
    assert set(empty) == set(broken) == set(to_report(None, None, None))
    assert failure_codes(None) == []
    assert failure_codes({"activeManifest": {"failure": [{"code": ""}, "x", {"code": "a"}]}}) == ["a"]


def test_post_c2pa_answers_each_image_and_refuses_a_bad_part(monkeypatch):
    import argus_retina.c2pa as c2pa

    original = c2pa.check
    resident = Resident("C2PA", lambda: object(), {"model": "c2pa-python"})
    resident.load()
    monkeypatch.setattr(c2pa, "resident", resident)
    monkeypatch.setattr(c2pa, "check", lambda _data: absent())
    client = TestClient(c2pa.app)
    files = [("f1", ("a.png", png(), "image/png")), ("f2", ("b.png", png(), "image/png"))]
    body = client.post("/c2pa", files=files).json()
    assert body["success"] is True and body["schema"] == SCHEMA and body["reports"] == [absent(), absent()]
    assert client.post("/c2pa", data={"x": "1"}).status_code == 400
    monkeypatch.setattr(c2pa, "check", original)
    bad = client.post("/c2pa", files=[("f1", ("a.png", b"not an image", "image/png"))])
    assert bad.status_code == 422


def test_an_unsigned_png_is_present_false_and_a_bmp_is_an_error():
    pytest.importorskip("c2pa")
    from argus_retina.c2pa import read_bytes

    assert read_bytes(png()) == absent()
    # BMP decodes as an image and has no C2PA container. The library refuses the type, so this is not a clean absence.
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    buf = io.BytesIO()
    image.save(buf, format="BMP")
    report = read_bytes(buf.getvalue())
    assert report["present"] is False and report["validation_state"] is None
    assert report["error"] and "NotSupported" in report["error"]
