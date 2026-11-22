# CLAUDE.md — argus-retina

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

The suite's **shared detector servers**: small, stateless model servers that take an image (or text) and return scores or detections. Each model is its own container image; all of them speak one HTTP contract. A retina senses and does not decide: thresholds, verdicts and selection stay in the calling stages.

Designed in [argus-cortex#88](https://github.com/smk762/argus-cortex/issues/88); its first job is [argus-cortex#79](https://github.com/smk762/argus-cortex/issues/79), one shared NudeNet 640m in place of the six copies across the suite.

**Status: phase 1 released (v0.1.0) and its callers migrated (#1); phase 2, SingGuard, built (#9); phase 3, faces, started (#10).** NudeNet 640m, EraX and watermark have moved here, OCR is new (#5), and SingGuard and faces are the GPU images. The watermark, SingGuard and faces images ship without weights (see the table below).

## Layout

```
src/argus_retina/  sidecar      # the contract: Resident (background load, one inference at a time, /health; on a GPU, /admin/*), create_app
                   nudenet erax watermark ocr singguard faces  # one module per model (yolo: the shared Ultralytics box conversion): load() imports the detector library lazily, so tests run without it
                   guard        # SingGuard's request shaping, prompt-injection guards and generation parsing, without torch
                   healthcheck  # `python -m argus_retina.healthcheck`: the images' HEALTHCHECK (healthy once loaded)
images/<model>/    Dockerfile requirements.txt gpu-tenant.yaml   # build from the repo root; weights/ is fetched, gitignored
                   # (images/watermark/weights/ is a runtime mount, dockerignored too: never in a build)
```

- `make weights` fetches NudeNet's 640m release asset with `gh` (anonymous GitHub downloads get a login page) and checks its sha256; the pin is in the Makefile **and** the Dockerfile (a test keeps them equal). CI and the release run it with the workflow token.
- Images get `VERSION` and `COMMIT` as build args (`.git` is not in the context); `COMMIT` becomes `/health`'s `commit`.

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/dev/SUITE.md

## Licence: AGPL-3.0 (whole repo)

Decided on cortex#88. The servers import AGPL code (`nudenet`, and `ultralytics` for EraX and watermark), so the repo is `AGPL-3.0-only`, one `LICENSE`, no per-image split.

- **Callers stay MIT** because they only talk HTTP. Never publish a retina module for other repos to import. The clients live in `argus_cortex.moderation` (`NudeNetClient`, `SingGuardClient`) and must never import anything from here.
- AGPL §13: build every published image from a tagged commit of this public repo, and have `/health` name its source commit.
- **`copier update` rewrites `LICENSE` and the `pyproject.toml` licence fields back to the template's MIT.** Reject those hunks every time.

## Scope

In: stateless perception models with no policy. Out:
- Thresholds, verdicts, selection: those belong to the stages (curator, darkroom, proof...).
- Image generation: darkroom fronts it, and voluptas hosts ComfyUI and diffusers.
- LLMs, speech and embeddings: agent-composer hosts them.

## Plan (from cortex#88)

1. **Phase 1: NudeNet 640m, EraX, watermark.** Move argus-cull's sidecars (`nudenet/`, `erax/`, `watermark/`, and their shared `sidecar-common/sidecar.py`) here. `sidecar.py` becomes the shared base in `src/argus_retina/`, and each model gets a directory with its own Dockerfile and `app.py`. Then switch the release from one image to one per model (`ghcr.io/smk762/argus-retina-<model>`). The migration issues are already filed: argus-curator#156, argus-darkroom#54, argus-proof#80, argus-studio#253, imogen#40, playactai/playact-engine#345, voluptas#25. Each needs the published image and its URL.
2. **Phase 2: SingGuard**, from argus-cull's `singguard/`, so cull and agent-composer stop each running their own. Check the licence of `inclusionAI/Sing-Guard-4b` first.
3. **Phase 3: faces** (InsightFace `buffalo_l`, MiVOLO), served here while curator keeps the decisions. `POST /faces` is the perception (boxes, embeddings, pose, age). `buffalo_l` is mounted, never baked in. FairFace and the person detector stay in curator. Callers are not migrated yet.

## The contract every image serves

- `POST /infer`: multipart image fields `f1`, `f2`... answered by `{"prediction": [[{"class", "score", "box": [x, y, w, h]}, ...], ...], "success": true}`, in field order. This is NudeNet's shape, already shared by cull's sidecars and parsed by `argus_cortex.moderation`. The OCR image adds a `text` field to each detection (`class` is `"text"`); keep new fields additive like that.
- `GET /health`: `argus_cortex.gpu_wire.ResidentHealth` (`model_state`, `busy`, `idle_s`, optional `gpu`), plus `model` and `resolution` (read by `NudeNetClient.info()`) and the source commit.
- `/admin/unload|evict|wake` (`ResidentActionReply`; 409 `ResidentBusyReply`) only on images that hold VRAM.
- A `gpu-tenant.yaml` per image declaring its footprint. Placement is by VRAM across 2–3 headless GPU workers, and argus-iris does the placing.

## Weights: what may be baked into a public image

Publishing to GHCR redistributes the weights inside each image. Pin every file by URL at a Hub commit **and** sha256, as cull's Dockerfiles already do.

| Model | Weights | Status |
|---|---|---|
| NudeNet 640m | NudeNet v3.4 release (cull currently fetches it from a third-party Hugging Face mirror) | AGPL, so OK to ship. Fetch from NudeNet's own release or a copy we pin; pinned at sha256 `04fe3d77…34eb` (the mirror's `5fd488c3…5c9e` is the same weights with the ONNX IR version lowered). |
| EraX-NSFW-V1.0 (s) | Apache-2.0 | OK |
| OCR | PP-OCRv6 small det/rec + PP-OCRv4 direction classifier, RapidOCR's v3.10.0 ONNX exports from ModelScope (Apache-2.0, PaddleOCR) | OK; pinned by release URL and sha256 in the Dockerfile |
| SingGuard-4b | `inclusionAI/SingGuard-4b` (was `Sing-Guard-4b`) at Hub commit `4e082d0`, on Qwen3-VL-4B-Instruct; both Apache-2.0 (checked on #9) | OK to ship, but 8.9 GB: **fetched at runtime** into the `/models/singguard` volume, every file checked against the sha256s in `singguard.FILES` (a test checks the Dockerfile fetches nothing). No remote code: transformers 4.57 has Qwen3-VL |
| Faces | InsightFace `buffalo_l` (non-commercial: **never bake it in**) mounted at `/models/insightface`; MiVOLO v2 ONNX (Apache-2.0, from `argus-curator mivolo-convert`) mounted at `/models/faces/mivolo_v2_d1_384.onnx` | A test checks the Dockerfile fetches nothing |
| Watermark | fancyfeast's YOLO11x (Space commit `9bb8f6a`, sha256 `80b022e9…326e`; pinned in the Makefile for `make watermark-weights`) | **No licence: never bake it in.** Mounted at `/models/watermark.pt`; a test checks the Dockerfile fetches nothing. mnemic's CC-BY-4.0 YOLOv8s lost the trial (15/32 vs 28/32 at 0.4, cortex#88) |

## Commands

```bash
make install   # uv venv + editable install with the "dev" extras
make test      # pytest --tb=short -q
make lint      # ruff check + format --check (pinned ruff 0.15.16)
make format    # ruff format + --fix
```

## Conventions & gotchas

- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `src/argus_retina/_version.py` is generated (gitignored). Tag `vX.Y.Z` to release.
- **Branches:** feature PRs target `dev`; `main` moves only by merging `dev`.
- `src/` layout, `py.typed`, Pydantic v2, `structlog`. Ruff line-length 120.
- Detector images run CPU ONNX/torch unless they say otherwise. Keep `CUDA_VISIBLE_DEVICES=` on CPU images, so they never claim a GPU. SingGuard and faces are the GPU images: `Resident(device="gpu")`, the admin routes (`argus_cortex.auth.bearer`, so they install argus-cortex), and a `gpu-tenant.yaml` resident with `unload_path`/`warm_path` (no `evict_path`: neither has a CPU stage). Faces caps each ONNX CUDA arena (`FACES_GPU_MEM_LIMIT_MB`); the arena is not returned on unload, so iris stops the container.
- SingGuard's caller text and policy go through `guard.neutralize` before the chat template: the tokenizer turns a written `<|im_end|>` into the control token, and the template writes turns as `[user]:` lines. Keep that for any new text field.

## CI / release

CI runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow. CI also builds every image (no push) on each PR. A `v*` tag pushes one image per model (`ghcr.io/smk762/argus-retina-<model>`) and then creates the GitHub Release from the tag's CHANGELOG section; it fails before pushing if that section is missing. **Nothing goes to PyPI**: nothing outside this repo should import it. `copier update` would bring back the template's PyPI release and single root Dockerfile: reject those hunks too. Scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
