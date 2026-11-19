# Changelog

All notable changes to this project are documented here. Versions come from git tags (`hatch-vcs`). A release's
section here becomes its GitHub Release notes, and the release fails before pushing images if the tag has none.

## [Unreleased]

## [0.1.1] - 2026-11-19

### Fixed

- **Audit findings (#8).** `decode_bgr` turns an empty or corrupt part (`cv2.error`) into `UnreadableImage` (422)
  instead of a 500. Decoded images are capped at `ARGUS_RETINA_MAX_PIXELS` (default 2**27) through
  `OPENCV_IO_MAX_IMAGE_PIXELS`, set before cv2 is imported, plus a check after decoding. The watermark weights guard
  joins continuation lines and checks every instruction, case-insensitively. The nudenet and ultralytics imports
  carry a reasoned `nosemgrep: agpl-import` marker, and a test keeps the markers and the AGPL licence field.

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
