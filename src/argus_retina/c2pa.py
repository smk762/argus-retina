"""C2PA manifest reader (CPU): ``uvicorn argus_retina.c2pa:app``.

``POST /c2pa`` reads a Content Credentials manifest from the original bytes and reports what the signer claimed,
plus the ``c2pa-python`` library's validation state. It does not decide whether the image is real or generated.
No manifest is ``present: false``: the file is unsigned, which is what most files are. A state of ``Invalid`` means
the signature or the content hash failed; ``failures`` names the library's codes for the active manifest.

The library uses its own default trust list. ``Trusted`` means the signature chains to an anchor on that list;
``Valid`` means the signature and the hash check without that. Nothing here adds or removes an anchor.

``c2pa-python`` is imported in ``load()``, so the tests import this module without it.
"""

from __future__ import annotations

from typing import Any

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

SCHEMA = "argus-retina.c2pa/v1"
# Both revisions of the actions assertion carry digitalSourceType on a created or edited action.
ACTION_LABELS = frozenset({"c2pa.actions", "c2pa.actions.v2"})


def absent() -> dict[str, Any]:
    """A file with no manifest. ``error`` stays null: the reader looked, and there was nothing to check."""
    return {
        "present": False,
        "validation_state": None,
        "failures": [],
        "claim_generator": None,
        "digital_source_types": [],
        "actions": [],
        "ingredients": 0,
        "signature": None,
        "error": None,
    }


def unreadable(exc: BaseException) -> dict[str, Any]:
    """The library could not read a store (an unsupported container, a broken manifest). Not a server failure."""
    report = absent()
    report["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return report


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _source_types(action: dict[str, Any]) -> list[str]:
    found: list[str] = []
    src = action.get("digitalSourceType")
    if isinstance(src, str) and src:
        found.append(src)
    params = action.get("parameters")
    if isinstance(params, dict):
        nested = params.get("digitalSourceType")
        if isinstance(nested, str) and nested:
            found.append(nested)
    return found


def claims(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Action names, in order, and the digital source types those actions declare."""
    actions: list[str] = []
    sources: list[str] = []
    for assertion in manifest.get("assertions") or []:
        if not isinstance(assertion, dict) or assertion.get("label") not in ACTION_LABELS:
            continue
        data = assertion.get("data")
        if not isinstance(data, dict):
            continue
        for action in data.get("actions") or []:
            if not isinstance(action, dict):
                continue
            name = action.get("action")
            if isinstance(name, str) and name:
                actions.append(name)
            sources.extend(_source_types(action))
    return actions, _dedupe(sources)


def failure_codes(results: Any) -> list[str]:
    """Codes from the active manifest's ``failure`` list. Other manifests in the store are not this file's verdict."""
    if not isinstance(results, dict):
        return []
    active = results.get("activeManifest")
    if not isinstance(active, dict):
        return []
    codes: list[str] = []
    for item in active.get("failure") or []:
        if isinstance(item, dict) and isinstance(item.get("code"), str) and item["code"]:
            codes.append(item["code"])
    return codes


def signature_of(info: Any) -> dict[str, str | None] | None:
    if not isinstance(info, dict):
        return None
    issuer = info.get("issuer") if isinstance(info.get("issuer"), str) else None
    when = info.get("time") if isinstance(info.get("time"), str) else None
    if issuer is None and when is None:
        return None
    return {"issuer": issuer, "time": when}


def to_report(manifest: Any, state: Any, results: Any) -> dict[str, Any]:
    """One active manifest as the fields a caller can record. The rest of the store (thumbnails, certificates) stays out."""
    report = absent()
    if not isinstance(manifest, dict):
        return report
    actions, sources = claims(manifest)
    generator = manifest.get("claim_generator")
    ingredients = manifest.get("ingredients")
    report.update(
        present=True,
        validation_state=state if isinstance(state, str) and state else None,
        failures=failure_codes(results),
        claim_generator=generator if isinstance(generator, str) and generator else None,
        digital_source_types=sources,
        actions=actions,
        ingredients=len(ingredients) if isinstance(ingredients, list) else 0,
        signature=signature_of(manifest.get("signature_info")),
    )
    return report


def read_bytes(data: bytes) -> dict[str, Any]:
    """The manifest in *data*, or an absent report. A library error is ``error`` on that report, not a raised failure."""
    import io  # noqa: PLC0415

    from c2pa import C2paError, Reader  # noqa: PLC0415 - only in the c2pa image

    try:
        reader = Reader.try_create(io.BytesIO(data))
    except C2paError as exc:
        return unreadable(exc)
    if reader is None:
        return absent()
    try:
        with reader:
            manifest = reader.get_active_manifest()
            state = reader.get_validation_state()
            results = reader.get_validation_results()
    except C2paError as exc:
        return unreadable(exc)
    return to_report(manifest, state, results)


def check(data: bytes) -> dict[str, Any]:
    """Decode first, so a part that is not an image is a 422, then read the original bytes (decoding drops the manifest)."""
    decode_bgr(data)
    return read_bytes(data)


def load() -> object:
    from c2pa import Reader  # noqa: PLC0415 - only in the c2pa image

    Reader.get_supported_mime_types()
    return object()


resident = Resident("C2PA", load, {"model": "c2pa-python", "schema": SCHEMA})
configure_logging()
app = create_app(resident, infer=False)


async def c2pa_route(request: Request) -> dict[str, Any]:
    if not resident.accepting:
        raise HTTPException(status_code=503, detail=f"{resident.name} is {resident.state}")
    async with request.form() as form:
        images = await form_images(form)
    if not images:
        raise HTTPException(status_code=400, detail="expected multipart image fields (f1, ...)")
    try:
        reports = await run_in_threadpool(resident.run, lambda _ready: [check(data) for data in images])
    except UnreadableImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        resident.log.exception("c2pa read failed")
        raise HTTPException(status_code=500, detail="C2PA read failed") from exc
    return {"schema": SCHEMA, "reports": reports, "success": True}


app.add_api_route("/c2pa", c2pa_route, methods=["POST"])
