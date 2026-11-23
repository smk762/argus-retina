"""SingGuard-4b (GPU): ``uvicorn argus_retina.singguard:app``. A policy verdict on an image, text, or both.

``POST /api/guard/check`` takes the JSON ``argus_cortex.moderation.SingGuardClient`` sends, ``{input, image_base64,
policy, thinking}`` or, for a text-only check, ``{input, input_role, policy, thinking}``, plus agent-composer's
``conversation`` (earlier turns, so a reply is judged in context). It answers ``{safe, verdict, rule, explanation,
categories: [{code, label, why}], fast_verdict, model, thinking, elapsed_ms}``. ``thinking`` is ``fast``,
``fast-slow``, ``slow`` or ``cascade`` (see ``argus_retina.guard``); a cascade reports the pass that decided, and
when slow decided, ``fast_verdict`` and ``fast_rule`` say what fast flagged. Policy texts and what to do with a
verdict stay with the callers.

The weights are not in the image. On the first load they are fetched from the Hub at a pinned commit into
``SINGGUARD_DIR`` (a volume; 8.9 GB), and every file is checked against the sha256 pinned below before it is used;
later loads read the volume and need no network. Moved from argus-cull's ``singguard/`` (argus-retina#9).

Settings: ``SINGGUARD_DIR`` (``/models/singguard``), ``SINGGUARD_QUANT`` (``4bit``, about 3.5 GB of VRAM, or
``bf16``, about 9 GB), ``SINGGUARD_MAX_NEW_TOKENS`` (2048, the reasoning modes' budget: a generation cut off before
its ``<answer>`` is a failed check), ``SINGGUARD_MAX_NEW_TOKENS_FAST`` (256), ``SINGGUARD_FORCE_THINKING`` (judge
every request in one mode, whatever it asks for). GPU leases come from the host-wide ``ARGUS_GPU_*`` settings
(``argus_cortex.gpu.coordinator_from_env``: ``ARGUS_GPU_BROKER_URL`` asks argus-iris, unset takes none); the tenant
is named by ``images/singguard/gpu-tenant.yaml``, copied into the image at ``/app/gpu-tenant.yaml``, and registered
with the broker at startup. The weights are Apache-2.0 (SingGuard-4b and its base,
Qwen3-VL-4B-Instruct).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from argus_cortex import settings
from fastapi import HTTPException
from pydantic import BaseModel, Field

from argus_retina import guard
from argus_retina.sidecar import NotReady, Resident, configure_logging, create_app

MODEL_ID = "inclusionAI/SingGuard-4b"  # was inclusionAI/Sing-Guard-4b; the old name redirects
# The Hub commit (head since 2026-06-23, as argus-cull pinned it) and each file's sha256 at it. The files are what
# transformers loads; the README and assets are left out.
REVISION = "4e082d0663ef44a44fc4d31c045294566d940e73"
FILES = {
    "added_tokens.json": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    "chat_template.jinja": "562724eed9009000c96cce9a725896b2a8ae4ebb774602d8b19aec68eb2cf33e",
    "config.json": "363e742cbeb2029a3afc66245788971c1c3ac6deb812082d010cb3eb4153bd4e",
    "generation_config.json": "4d9818c3d27895c0058828a5f68bc7a4de80c3ae1bcdac90936ce24178063f59",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00002.safetensors": "3eb66ca8f7df2979c69fa68a6806c49d37383fa1f6a515016b57baf16e2bb9c1",
    "model-00002-of-00002.safetensors": "cc74b95fce683cf7a012d68494eefe08ace08a6c59b73a86a8b9031e7a60018e",
    "model.safetensors.index.json": "c321e968083cc64dbdc77b9cc17842396cf376399f4ff4bca6b0a001fd22afef",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "9f42f5fc068755b2d606e95d97a4e3b39f363c21acfb37bdcfb12a1c003e7967",
    "video_preprocessor_config.json": "59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
# Written beside the files once every one has matched, so a restart doesn't hash 8.9 GB again.
VERIFIED = ".argus-retina-verified"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return value if value > 0 else default


DIR = Path(os.environ.get("SINGGUARD_DIR", "/models/singguard"))
QUANT = os.environ.get("SINGGUARD_QUANT", "4bit").strip() or "4bit"
if QUANT not in ("4bit", "bf16"):
    raise SystemExit(f"SINGGUARD_QUANT must be 4bit or bf16, got {QUANT!r}")
MAX_NEW = {
    "fast": _int_env("SINGGUARD_MAX_NEW_TOKENS_FAST", 256),
    "fast-slow": _int_env("SINGGUARD_MAX_NEW_TOKENS", 2048),
    "slow": _int_env("SINGGUARD_MAX_NEW_TOKENS", 2048),
}
# For trying slow or cascade from a client that can only ask for fast or fast-slow.
FORCE_THINKING = os.environ.get("SINGGUARD_FORCE_THINKING", "").strip() or None
if FORCE_THINKING is not None and FORCE_THINKING not in guard.THINKING:
    raise SystemExit(f"SINGGUARD_FORCE_THINKING must be one of {', '.join(guard.THINKING)}, got {FORCE_THINKING!r}")

# The lease's footprint, as images/singguard/gpu-tenant.yaml declares it for 4-bit (tests keep the two equal): the whole
# check (weights, a 1536 px image and a 2048-token write-up) and the part that stays loaded after it. bf16 is larger.
FOOTPRINT_MB = {"4bit": (6000, 3600), "bf16": (12000, 9000)}

# Text bounds: every character is prompt tokens on the GPU. A policy runs to a few thousand characters.
MAX_TEXT = 32_768
MAX_TURNS = 32


# --- weights -------------------------------------------------------------------------------------------------------


def _manifest() -> str:
    return json.dumps({"repo": MODEL_ID, "revision": REVISION, "files": FILES}, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> list[str]:
    """The pinned files under *root* that are missing or don't match their sha256."""
    bad = []
    for name, want in FILES.items():
        path = root / name
        if not path.is_file() or _sha256(path) != want:
            bad.append(name)
    return bad


def weights(base: Path = DIR) -> Path:
    """The pinned SingGuard files, checked: from *base* when they're there, else fetched into it first.

    Raises RuntimeError when a file still doesn't match after fetching (the bad files are removed, so the next load
    fetches them again).
    """
    root = base / REVISION
    marker = root / VERIFIED
    if marker.is_file() and marker.read_text() == _manifest():
        return root
    if verify(root):
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - in the singguard image

        resident.log.info("fetching %s@%s into %s", MODEL_ID, REVISION, root)
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=root, allow_patterns=list(FILES))
        bad = verify(root)
        if bad:
            for name in bad:
                (root / name).unlink(missing_ok=True)
            raise RuntimeError(f"{MODEL_ID}@{REVISION[:7]}: sha256 mismatch for {', '.join(bad)}")
    marker.write_text(_manifest())
    return root


# --- the model -----------------------------------------------------------------------------------------------------


class Guard:
    """The loaded processor and model: one generation per call (the resident runs one at a time)."""

    def __init__(self, model: Any, processor: Any, torch: Any):
        self.model, self.processor, self.torch = model, processor, torch

    def generate(self, chat: list[dict[str, Any]], thinking: str, policy: str | None) -> str:
        prompt = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **guard.template_kwargs(thinking, policy)
        )
        if thinking == "slow":
            prompt += guard.SLOW_PREFILL
        images = [part["image"] for message in chat for part in message["content"] if part["type"] == "image"]
        with self.torch.inference_mode():
            inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt").to(self.model.device)
            # Greedy: the Hub's generation_config samples, and a verdict must not depend on a dice roll.
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW[thinking], do_sample=False, temperature=None, top_p=None, top_k=None
            )
            trimmed = out[:, inputs["input_ids"].shape[-1] :]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Give the prefilled heading back so the recorded explanation is whole.
        return guard.SLOW_PREFILL + text if thinking == "slow" else text


def load() -> Guard:
    root = weights()
    import torch  # noqa: PLC0415 - in the singguard image
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: SingGuard runs on the GPU")
    kwargs: dict[str, Any] = {"device_map": {"": 0}}
    if QUANT == "4bit":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs["dtype"] = torch.bfloat16
    processor = AutoProcessor.from_pretrained(root)
    model = AutoModelForImageTextToText.from_pretrained(root, **kwargs).eval()
    return Guard(model, processor, torch)


_coordinator: Any = None
# The startup load's thread (the lease) and the lifespan (the registration) both ask for it: build it once.
_coordinator_lock = threading.Lock()


def coordinator() -> Any:
    """The host's GPU coordinator, built once: the leases and the broker registration share it."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            from argus_cortex.gpu import coordinator_from_env  # noqa: PLC0415 - in the singguard image

            _coordinator = coordinator_from_env()
        return _coordinator


def lease() -> Any:
    """A ``guard`` lease from the host's GPU coordinator, made on first use (the broker client is built once)."""
    vram_mb, model_mb = FOOTPRINT_MB[QUANT]
    return coordinator().lease(
        caller="argus-retina-singguard", min_vram_mb=vram_mb, model_mb=model_mb, kind="guard", priority="interactive"
    )


def lock_warning(env: Mapping[str, str] = os.environ) -> str | None:
    """Why the GPU lease's lock-file fallback would fail here, or None.

    The image runs as a non-root user, and Docker creates a missing bind-mount directory as root, mode 755, so the
    first lease that falls back to the host lock (the broker unreachable, say after a reboot) is refused. Said once
    at startup, naming the path, rather than only in the 503 of a later check (argus-retina#20).
    """
    # cortex's choice: ARGUS_GPU_COORDINATOR, else the broker when ARGUS_GPU_BROKER_URL is set.
    url = env.get("ARGUS_GPU_BROKER_URL", "").strip()
    coordinator = env.get("ARGUS_GPU_COORDINATOR", "").strip().lower() or ("broker" if url else "none")
    fallback = env.get("ARGUS_GPU_BROKER_FALLBACK", "lease").strip().lower() or "lease"
    if coordinator != "lease" and not (coordinator == "broker" and fallback == "lease"):
        return None
    path = env.get("ARGUS_GPU_LEASE_PATH", "").strip()
    if not path:
        from argus_cortex.gpu import default_lock_path  # noqa: PLC0415 - in the singguard image

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


def release() -> None:
    """After an unload: collect the dropped model, then hand its cached blocks back to the driver.

    cuBLAS's workspace (8 MB, one per thread that ran a generation) is allocated from torch's cache and outlives the
    model; it sits inside a segment the load left and pins it, so without clearing it an unload gave back almost nothing
    (4.2 GB stayed reserved, measured on a 3090). Cleared, the whole reservation returns.
    """
    gc.collect()
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)  # private, so guarded
        if clear is not None:
            clear()
        torch.cuda.empty_cache()


def memory() -> dict[str, float] | None:
    """``/health``'s ``gpu`` block (cortex's ``ResidentGpu``), once torch is loaded."""
    import sys  # noqa: PLC0415

    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    mb = 1024 * 1024
    return {
        "free_vram_mb": round(free / mb),
        "total_mb": round(total / mb),
        "allocated_mb": round(torch.cuda.memory_allocated() / mb),
        "reserved_mb": round(torch.cuda.memory_reserved() / mb),
    }


# --- the route -----------------------------------------------------------------------------------------------------


class GuardMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_TEXT)


class CheckRequest(BaseModel):
    input: str = Field(default="", max_length=MAX_TEXT)
    image_base64: str = ""
    input_role: Literal["user", "assistant"] = "user"
    conversation: list[GuardMessage] | None = Field(default=None, max_length=MAX_TURNS)
    policy: str | None = Field(default=None, max_length=MAX_TEXT)
    thinking: str = "fast"


def judge(g: Guard, chat: list[dict[str, Any]], policy: str | None, thinking: str) -> dict[str, Any]:
    """One generation in *thinking*, parsed. Raises ``guard.NoVerdict``."""
    started = time.perf_counter()
    generated = g.generate(chat, thinking, policy)
    try:
        response = guard.to_response(generated, MODEL_ID, int((time.perf_counter() - started) * 1000), policy)
    except guard.NoVerdict:
        # The tail says whether the budget ran out before the <answer> tag.
        resident.log.warning("no verdict (%s, %d chars); tail: %r", thinking, len(generated), generated[-160:])
        raise
    response["thinking"] = thinking
    return response


def cascade(g: Guard, chat: list[dict[str, Any]], policy: str | None) -> dict[str, Any]:
    """Fast, then slow for what fast flagged; slow's verdict stands, with ``fast_verdict`` and ``fast_rule`` kept.

    Fast over-flags rather than under-flags (its verdict is the first token fast-slow writes before reasoning), so it
    is a cheap screen: what it passes is not looked at again.
    """
    fast = judge(g, chat, policy, "fast")
    if fast["safe"]:
        return fast
    slow = judge(g, chat, policy, "slow")
    slow["fast_verdict"] = "unsafe"
    slow["fast_rule"] = fast["rule"]
    slow["elapsed_ms"] += fast["elapsed_ms"]
    return slow


def check(body: CheckRequest) -> dict[str, Any]:
    if body.thinking not in guard.THINKING:
        raise HTTPException(status_code=400, detail=f"thinking must be one of {', '.join(guard.THINKING)}")
    if not resident.accepting:  # a failed GPU load is tried again in run()
        raise HTTPException(status_code=503, detail=f"{resident.name} is {resident.state}")
    thinking = FORCE_THINKING or body.thinking
    try:
        image = guard.decode_image(body.image_base64) if body.image_base64.strip() else None
        turns = [(m.role, m.content) for m in body.conversation or []]
        chat = guard.messages(body.input, role=body.input_role, conversation=turns, image=image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def work(g: Guard) -> dict[str, Any]:
        if thinking == "cascade":  # both passes under one hold of the model
            return cascade(g, chat, body.policy)
        return judge(g, chat, body.policy, thinking)

    try:
        return resident.run(work)
    except NotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except guard.NoVerdict as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        resident.log.exception("guard check failed")
        raise HTTPException(status_code=500, detail="SingGuard check failed") from exc


resident = Resident(
    "SingGuard",
    load,
    {"model": MODEL_ID, "revision": REVISION, "quant": QUANT, "force_thinking": FORCE_THINKING},
    device="gpu",
    release=release,
    memory=memory,
    lease=lease,
)
configure_logging()
# The suite file behind the environment, as cortex reads the same ARGUS_GPU_* settings (argus_cortex.settings).
if (_lock_problem := lock_warning(settings.environ)) is not None:
    resident.log.warning("%s", _lock_problem)
app = create_app(resident, infer=False, coordinator=coordinator)
# A sync route: FastAPI runs it on the threadpool, so a generation never blocks /health on the event loop.
app.add_api_route("/api/guard/check", check, methods=["POST"])
