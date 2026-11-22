# Changelog

All notable changes to this project are documented here. Versions come from git tags (`hatch-vcs`). A release's
section here becomes its GitHub Release notes, and the release fails before pushing images if the tag has none.

## [Unreleased]

## [0.2.1] - 2026-11-22

### Fixed

- **A GPU model that failed to load tries again on the next request (#19, #20).** `/api/guard/check` and `/infer` refused
  with a `503` whenever the resident wasn't `ready`, so an out-of-memory load on a GPU image stayed in `error` until
  `/admin/wake` or a restart, even though `run()` retries it. They now let a GPU resident in `error` through to
  `run()` (`Resident.accepting`); `ready`, and with it the HEALTHCHECK, still reports the failure. A CPU image that
  failed to load stays a `503`.
- SingGuard logs a warning at startup when the GPU lease's lock-file fallback would be refused (the lock file can't
  be created or opened by the image's non-root user), naming the path, instead of only failing a check later.
- The README documents the lock-file fallback (#18): create `ARGUS_GPU_LOCK_DIR` mode 1777, with a tmpfiles.d rule so
  it survives reboots.

## [0.2.0] - 2026-11-22

### Added

- **SingGuard, phase 2 of argus-cortex#88 (#9): `ghcr.io/smk762/argus-retina-singguard`**, one SingGuard-4b for the
  suite in place of argus-cull's `singguard/` service and agent-composer's guard route. `POST /api/guard/check` takes
  what `argus_cortex.moderation.SingGuardClient` sends, so callers change only a URL: image checks, text-only checks
  (`input_role`), and agent-composer's `conversation`; `thinking` is `fast`, `fast-slow`, `slow` or `cascade`, as cull
  served them. Moved from cull: the parsing (an unreadable `<answer>` is unsafe with no rule; no verdict is a `500`,
  never a pass) and slow's prefilled first step. From composer: control tokens and `[role]:` markers in the caller's
  text and policy are made plain, so they can't forge a verdict or a turn. Images are shrunk to 1536 px first.
- **The weights are fetched, not baked in.** SingGuard-4b and its base are Apache-2.0, but 8.9 GB: the first start
  downloads them at a pinned Hub commit into the `/models/singguard` volume and checks every file's sha256; later
  starts need no network. 4-bit bitsandbytes by default (about 3.5 GB of VRAM), `SINGGUARD_QUANT=bf16` for full
  precision.
- **GPU residents.** `Resident(device="gpu")` reports `gpu` and its memory on `/health` and serves argus-cortex's
  `/admin/unload`, `/admin/evict` (an unload here) and `/admin/wake`, behind `ARGUS_ADMIN_TOKEN` when set, with `409`
  while a check runs. After an unload the next request loads the model again, and the HEALTHCHECK counts an unloaded
  GPU model as healthy. `images/singguard/gpu-tenant.yaml` declares the `guard` kind and the resident for argus-iris.
- **GPU leases.** SingGuard takes a `guard` lease from argus-iris's broker (`argus_cortex.gpu`, the host's
  `ARGUS_GPU_*` settings) around its startup load and around each check, including a reload after an unload, so iris
  can make room first. `/admin/wake` takes none: iris calls it after making room. A lease that can't be had is a `503`.
- `docker compose --profile gpu up -d` starts SingGuard on `:8134`.

### Changed

- The `dev` extra needs `argus-cortex>=0.8` (`argus_cortex.auth`) and Pillow.

### Fixed

- **Audit findings (#14).** SingGuard reads the first line's verdict as a whole word after any `Safety:` label, so
  `Safety: Unsafe` is unsafe and `safety concern` is no verdict (a prefix match on `safe` read both as safe). A reply
  about an image with no prompt keeps its assistant role, with the image on a user turn of its own. A refused startup
  lease leaves a GPU model unloaded rather than in `error`, and a GPU load that failed is tried again on the next
  request, so a busy GPU at boot no longer means a 503 until iris wakes it.

### Security

- SingGuard's image moves to transformers 5.22.2, accelerate 1.15.0 and Pillow 13.0.0, clearing the 66 Dependabot
  alerts on `images/singguard/requirements.txt` (Pillow decoder overflows and decompression-bomb bypasses,
  transformers and accelerate path traversal and code-execution advisories). Checked on a 3090 against the pinned
  weights: it loads in 4-bit, every thinking mode gives the expected verdicts, and unload and wake work.

## [0.1.1] - 2026-11-19

### Fixed

- **Audit findings (#8).** `decode_bgr` turns an empty or corrupt part (`cv2.error`) into `UnreadableImage` (422)
  instead of a 500. Decoded images are capped at `ARGUS_RETINA_MAX_PIXELS` (default 2**27) through
  `OPENCV_IO_MAX_IMAGE_PIXELS`, set before cv2 is imported, plus a check after decoding. The watermark weights guard
  joins continuation lines and checks every instruction, case-insensitively. The nudenet and ultralytics imports
  carry a reasoned `nosemgrep: agpl-import` marker, and a test keeps the markers and the AGPL licence field.
- **Release review (#11).** A non-positive or overflowing `ARGUS_RETINA_MAX_PIXELS` fails at import with a clear
  error instead of OpenCV aborting the process; an OpenCV out-of-memory error stays a 500, not a 422. The watermark
  weights guard drops comment lines before joining continuations and knows more weight formats and fetchers. The
  README lists `ARGUS_RETINA_MAX_PIXELS`.

## [0.1.0] - 2026-11-16

### Added

- **Phase 1 (argus-cortex#88, #79): NudeNet 640m and EraX-NSFW, moved from argus-cull's sidecars.** One image per
  model, `ghcr.io/smk762/argus-retina-nudenet` and `ghcr.io/smk762/argus-retina-erax`, on one shared server
  (`argus_retina.sidecar`, from cull's `sidecar-common`):
  - `POST /infer` in NudeNet's shape, as cull's sidecars served it. A file that isn't an image is now a `422`
    (it was a `500`), and `/infer` answers `503` while the model loads.
  - `GET /health` answers argus-cortex's `ResidentHealth` (`model_state`, `busy`, `idle_s`), plus the model and its
    settings (what `NudeNetClient.info()` reads) and the `version` and source `commit` (AGPL section 13). The model
    loads on a background thread, so `/health` answers `loading` rather than the port staying closed.
  - Each image runs as a non-root user on the CPU, has a HEALTHCHECK that waits for the model, and declares
    itself to argus-iris in `images/<model>/gpu-tenant.yaml` (CPU only, no VRAM).
- **NudeNet's own 640m weights**, sha256 `04fe3d77…34eb`, fetched with `gh` (`make weights`) because GitHub serves
  release assets to anonymous clients as a login page. Cull used a third-party mirror: the same weights with the ONNX
  IR version lowered from 10 to 9.
- **Watermark detector (argus-cortex#88), moved from argus-cull's `watermark/` sidecar:** `ghcr.io/smk762/argus-retina-watermark`, for
  curator's `crops.detect`. The image has **no weights**. Mount them at `/models/watermark.pt`: fancyfeast's YOLO11x states no licence,
  so it is never redistributed. `make watermark-weights` fetches it pinned by Space commit and sha256. mnemic's CC-BY-4.0 YOLOv8s
  was tried in its place and caught 15 of 32 watermarked images against the YOLO11x's 28 at curator's 0.4 cutoff. Boxes are whole pixels
  now, as EraX's are (cull's sidecar rounded them to 0.1).
- **OCR (#5): `ghcr.io/smk762/argus-retina-ocr`**, RapidOCR with PaddleOCR's PP-OCRv6 small (Apache-2.0, ONNX on the CPU),
  for checking the text a caption claims to read (argus-lens reconcile). `/infer` answers one detection per line of
  text, `{"class": "text", "score", "box", "text"}`, in reading order. The three models are pinned to RapidOCR's v3.10.0
  release by sha256 and never downloaded at runtime. Chosen over docTR, EasyOCR and Florence-2: on 31 photos with
  hand-checked text it found 48 of 69 phrases at about 0.4 s per image, and unlike Florence-2 it never wrote text
  that isn't there.
- `compose.yaml` runs all four on loopback (`:8130` to `:8133`). CI builds every image on every PR; a `v*` tag pushes
  them and creates the GitHub Release. Nothing goes to PyPI.
