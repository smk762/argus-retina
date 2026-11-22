"""The SingGuard image's server around a scripted stand-in model: the route, the cascade, the weights check.

No torch, transformers or network: ``Guard.generate`` is replaced by canned generations.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import sys
import types

import pytest
from argus_cortex.gpu_wire import ResidentHealth
from argus_cortex.moderation.singguard import to_result
from fastapi.testclient import TestClient
from PIL import Image

from argus_retina import singguard
from argus_retina.sidecar import Resident, create_app


class FakeGuard:
    """Answers each generation from *script* (by thinking mode) and records what it was asked."""

    def __init__(self, script: dict[str, str]):
        self.script = script
        self.calls: list[tuple[str, str | None, list]] = []

    def generate(self, chat, thinking, policy):
        self.calls.append((thinking, policy, chat))
        return self.script[thinking]


@pytest.fixture
def served(monkeypatch):
    """(client, guard): the route on a fresh GPU resident whose model is a FakeGuard."""
    fake = FakeGuard({})
    r = Resident("SingGuard", lambda: fake, {"model": singguard.MODEL_ID}, device="gpu")
    monkeypatch.setattr(singguard, "resident", r)
    monkeypatch.setattr(singguard, "FORCE_THINKING", None)
    app = create_app(r, load=False, infer=False)
    app.add_api_route("/api/guard/check", singguard.check, methods=["POST"])
    r.load()
    return TestClient(app), fake


def png() -> str:
    out = io.BytesIO()
    Image.new("RGB", (8, 6)).save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


SAFE = "safe\n<answer>Safe</answer>"
UNSAFE = "unsafe\n<answer>A. Sexual Content Risk</answer>"
SLOW_SAFE = "A portrait.\n[Step 3] Final Judgment\nSafe.\n<answer>Safe</answer>"
SLOW_UNSAFE = "Explicit.\n[Step 3] Final Judgment\nUnsafe.\n<answer>A. Sexual Content Risk</answer>"


def test_an_image_check_answers_what_the_cortex_client_reads(served):
    c, fake = served
    fake.script["fast"] = UNSAFE
    r = c.post("/api/guard/check", json={"input": "a prompt", "image_base64": png(), "thinking": "fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["thinking"] == "fast" and body["model"] == "inclusionAI/SingGuard-4b"
    result = to_result(body)  # argus_cortex.moderation's own parsing
    assert result.safe is False and result.rules == ["A. Sexual Content Risk"] and result.thinking == "fast"
    thinking, policy, chat = fake.calls[0]
    assert thinking == "fast" and policy is None
    assert chat[0]["content"][0]["type"] == "image" and chat[0]["content"][1]["text"] == "a prompt"


def test_a_text_only_check_and_a_conversation(served):
    c, fake = served
    fake.script["fast-slow"] = "safe\n[Step 1] Content Summary\nA question.\n<answer>Safe</answer>"
    body = {
        "input": "Sure, here's how",
        "input_role": "assistant",
        "conversation": [{"role": "user", "content": "how do I bake bread"}],
        "policy": "### A. Weapons\n  - guns",
        "thinking": "fast-slow",
    }
    r = c.post("/api/guard/check", json=body)
    assert r.status_code == 200 and to_result(r.json()).safe is True
    _, policy, chat = fake.calls[0]
    assert policy == "### A. Weapons\n  - guns"
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert all(part["type"] == "text" for m in chat for part in m["content"])


def test_cascade_passes_on_fast_alone(served):
    c, fake = served
    fake.script["fast"] = SAFE
    body = c.post("/api/guard/check", json={"image_base64": png(), "thinking": "cascade"}).json()
    assert body["safe"] is True and body["thinking"] == "fast" and [t for t, *_ in fake.calls] == ["fast"]


@pytest.mark.parametrize(("slow", "safe"), [(SLOW_SAFE, True), (SLOW_UNSAFE, False)])
def test_cascade_settles_what_fast_flagged_with_slow(served, slow, safe):
    c, fake = served
    fake.script.update(fast=UNSAFE, slow=guard_prefilled(slow))
    body = c.post("/api/guard/check", json={"image_base64": png(), "thinking": "cascade"}).json()
    assert [t for t, *_ in fake.calls] == ["fast", "slow"]
    assert body["safe"] is safe and body["thinking"] == "slow"
    assert body["fast_verdict"] == "unsafe" and body["fast_rule"] == "A. Sexual Content Risk"
    assert body["explanation"].startswith("[Step 1] Content Summary")


def guard_prefilled(text: str) -> str:
    # Guard.generate gives the prefilled heading back; the fake stands in for that.
    return "[Step 1] Content Summary\n" + text


def test_force_thinking_overrides_the_request(served, monkeypatch):
    c, fake = served
    monkeypatch.setattr(singguard, "FORCE_THINKING", "slow")
    fake.script["slow"] = guard_prefilled(SLOW_SAFE)
    body = c.post("/api/guard/check", json={"image_base64": png(), "thinking": "fast"}).json()
    assert body["thinking"] == "slow"


@pytest.mark.parametrize(
    "body",
    [
        {"image_base64": png(), "thinking": "deep"},
        {"image_base64": "not an image"},
        {"input": "   "},
        {"input": "x", "conversation": [{"role": "user", "content": "y"}] * (singguard.MAX_TURNS + 1)},
    ],
    ids=["thinking", "image", "nothing", "turns"],
)
def test_a_bad_request_is_a_4xx_and_reaches_no_model(served, body):
    c, fake = served
    assert c.post("/api/guard/check", json=body).status_code in (400, 422)
    assert fake.calls == []


def test_no_verdict_is_a_500_not_a_pass(served):
    c, fake = served
    fake.script["fast"] = "I am not sure"
    r = c.post("/api/guard/check", json={"image_base64": png()})
    assert r.status_code == 500 and "no verdict" in r.json()["detail"]


def test_a_model_failure_is_a_500_without_the_detail(served):
    c, fake = served
    fake.script = {}  # KeyError inside generate
    r = c.post("/api/guard/check", json={"image_base64": png()})
    assert r.status_code == 500 and r.json()["detail"] == "SingGuard check failed"


def test_a_check_after_an_unload_loads_the_model_again(served):
    c, fake = served
    fake.script["fast"] = SAFE
    assert c.post("/admin/unload").json()["model_state"] == "unloaded"
    assert ResidentHealth.model_validate(c.get("/health").json()).model_state == "unloaded"
    assert c.post("/api/guard/check", json={"image_base64": png()}).json()["safe"] is True
    assert c.get("/health").json()["model_state"] == "gpu"


def test_the_module_serves_the_guard_route_and_the_admin_routes_and_names_its_pin():
    assert singguard.resident.device == "gpu" and singguard.resident.state == "unloaded"
    assert singguard.resident.info["revision"] == singguard.REVISION
    assert {r.path for r in singguard.app.routes} >= {"/health", "/api/guard/check", "/admin/unload", "/admin/wake"}
    assert "/infer" not in {r.path for r in singguard.app.routes}


# --- weights -------------------------------------------------------------------------------------------------------


@pytest.fixture
def pinned(monkeypatch):
    """Two small pinned files, and a fake huggingface_hub whose download writes *served* into the target dir."""
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    monkeypatch.setattr(singguard, "FILES", {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    served = dict(files)
    downloads = []

    def snapshot_download(repo, *, revision, local_dir, allow_patterns):
        downloads.append((repo, revision, sorted(allow_patterns)))
        local_dir.mkdir(parents=True, exist_ok=True)
        for name, data in served.items():
            (local_dir / name).write_bytes(data)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return served, downloads


def test_weights_are_fetched_once_at_the_pinned_revision_and_checked(pinned, tmp_path):
    _, downloads = pinned
    root = singguard.weights(tmp_path)
    assert root == tmp_path / singguard.REVISION
    assert downloads == [(singguard.MODEL_ID, singguard.REVISION, ["config.json", "model.safetensors"])]
    assert singguard.weights(tmp_path) == root and len(downloads) == 1  # verified: no second fetch


def test_weights_that_dont_match_are_removed_and_refused(pinned, tmp_path):
    served, _ = pinned
    served["model.safetensors"] = b"a login page"
    with pytest.raises(RuntimeError, match="sha256 mismatch for model.safetensors"):
        singguard.weights(tmp_path)
    root = tmp_path / singguard.REVISION
    assert not (root / "model.safetensors").exists() and not (root / singguard.VERIFIED).exists()


def test_a_changed_pin_checks_the_files_again(pinned, tmp_path, monkeypatch):
    _, downloads = pinned
    singguard.weights(tmp_path)
    monkeypatch.setitem(singguard.FILES, "config.json", "0" * 64)
    with pytest.raises(RuntimeError, match="config.json"):
        singguard.weights(tmp_path)
    assert len(downloads) == 2


def test_every_file_transformers_loads_is_pinned():
    names = set(singguard.FILES)
    assert {"config.json", "chat_template.jinja", "tokenizer.json", "model.safetensors.index.json"} <= names
    assert {n for n in names if n.endswith(".safetensors")} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in singguard.FILES.values())
    assert len(singguard.REVISION) == 40


def test_the_lease_footprint_is_the_manifests_and_comes_from_the_hosts_coordinator(monkeypatch):
    from pathlib import Path

    from argus_cortex.gpu_wire import load_tenant_manifest

    m = load_tenant_manifest(Path(__file__).parent.parent / "images" / "singguard" / "gpu-tenant.yaml")
    (kind,) = m.kinds
    assert singguard.FOOTPRINT_MB["4bit"] == (kind.vram_mb, kind.model_mb)

    asked = []

    class Coordinator:
        def lease(self, **kwargs):
            asked.append(kwargs)
            return contextlib.nullcontext()

    import argus_cortex.gpu

    monkeypatch.setattr(singguard, "_coordinator", None)
    monkeypatch.setattr(argus_cortex.gpu, "coordinator_from_env", Coordinator)
    with singguard.lease():
        pass
    assert asked == [
        {
            "caller": "argus-retina-singguard",
            "min_vram_mb": kind.vram_mb,
            "model_mb": kind.model_mb,
            "kind": "guard",
            "priority": "interactive",
        }
    ]
