# argus-retina

The Argus suite's shared detector servers. Each server is a small, stateless model: an image goes in, and scores or detections come out. Each model ships as its own container image, and all of them speak one HTTP contract. Retina only senses. The calling stages own the thresholds and verdicts.

Part of the [Argus suite](https://github.com/smk762/argus-cortex/blob/dev/SUITE.md). Design: [argus-cortex#88](https://github.com/smk762/argus-cortex/issues/88). First job: [argus-cortex#79](https://github.com/smk762/argus-cortex/issues/79), one shared NudeNet 640m instead of a copy in every repo.

## Images

| Image | Model | Weights licence | Runs on |
|---|---|---|---|
| `ghcr.io/smk762/argus-retina-nudenet` | [NudeNet](https://github.com/notAI-tech/NudeNet) **640m** (v3.4 weights) at 640 px | AGPL-3.0 (NudeNet) | CPU (ONNX) |
| `ghcr.io/smk762/argus-retina-erax` | [EraX-NSFW V1.0](https://huggingface.co/erax-ai/EraX-NSFW-V1.0) small (YOLO11s) | Apache-2.0 | CPU (torch) |
| `ghcr.io/smk762/argus-retina-watermark` | Watermark detector, **no weights in the image**: mount fancyfeast's [YOLO11x](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection) | none stated, so never redistributed | CPU (torch) |
| `ghcr.io/smk762/argus-retina-ocr` | [RapidOCR](https://github.com/RapidAI/RapidOCR) with PaddleOCR's **PP-OCRv6 small** (multilingual Latin and CJK): text lines with boxes | Apache-2.0 | CPU (ONNX) |

All listen on port 8080, run as a non-root user, never claim a GPU (`CUDA_VISIBLE_DEVICES` is empty), and are healthy once their model has loaded.

```bash
make watermark-weights        # the YOLO11x the watermark service mounts (pinned by Space commit and sha256)
docker compose up -d          # on 127.0.0.1:8130 (nudenet), :8131 (erax), :8132 (watermark) and :8133 (ocr)
```

### OCR: what the image says

The OCR image reads the text in an image so a caller can check what a caption claims it reads (argus-lens reconcile, [#5](https://github.com/smk762/argus-retina/issues/5)). It returns one detection per line of text, with the line in `text`, in reading order: rows top to bottom, then left to right. It reads rows, not columns, so two facing pages interleave. `score` is the recogniser's confidence; a line it can barely read (ornate script, Fraktur) has a low score rather than a cleaned-up guess. It reads what is printed, misspellings included. Words are sometimes run together (`afProfessor`), so a name check should compare without spaces.

It was chosen over docTR, EasyOCR and Florence-2 on 31 photos with hand-checked text (historical documents and plates, street signs). RapidOCR found 48 of 69 phrases at about 0.4 s per image on the CPU. Florence-2 large found 51 but took 8 s, and it wrote plausible text that isn't in the image ("Hosibolt Maria Espinosa" for Josep Maria Espinàs), which is what a text check exists to catch.

### Watermark: bring your own weights

The watermark image serves curator's `crops.detect` and ships without a model. Mount one at `/models/watermark.pt`; compose mounts `images/watermark/weights/watermark.pt`, or the file `WATERMARK_WEIGHTS` names. Without it the server stays up, and `/health` reports `error` with the reason.

The suite runs fancyfeast's YOLO11x, the detector behind JoyCaption's watermark tags. It states no licence, so no published image contains it. mnemic's CC-BY-4.0 YOLOv8s, which could have been baked in, was tried instead. On 32 watermarked images it caught 15 against the YOLO11x's 28 at curator's 0.4 cutoff, and its boxes rarely matched ([cortex#88](https://github.com/smk762/argus-cortex/issues/88#issuecomment-6913605897)). Any Ultralytics detector works if its classes suit the caller.

### NudeNet 640m, not 320n

The upstream `ghcr.io/notai-tech/nudenet` image and the `nudenet` wheel bundle only the 320n detector. 640m finds more, and the two score the same image differently, so **a threshold tuned on one does not carry over to the other**. Check your cutoffs when you move from 320n. `NudeNetClient.info()` reports which detector a server runs.

The 640m file is NudeNet's own release asset (sha256 `04fe3d77…34eb`). GitHub serves release assets to an anonymous client as a login page, so `make weights` fetches it with `gh`. argus-cull's sidecar used a third-party Hugging Face mirror of it: the same weights, with the ONNX IR version in the header lowered from 10 to 9.

## The contract

- `POST /infer`: multipart image fields `f1`, `f2`, …, answered by `{"prediction": [[{"class", "score", "box": [x, y, w, h]}, ...], ...], "success": true}`, one list per image in field order. This is NudeNet's shape; EraX answers in it with its own classes (`anus`, `make_love`, `nipple`, `penis`, `vagina`), and the watermark detector with `watermark`. The OCR image answers with `text`, and each of its detections adds the line it read as a `text` field. `400` without images, `422` for a file that isn't an image, `503` while the model loads.
- `GET /health`: argus-cortex's `ResidentHealth` (`model_state`: `loading`, `cpu` or `error`; `busy`; `idle_s`), plus `detector`, `model` (the weights file) and the model's settings (`resolution` for NudeNet, `conf`/`iou` for EraX, `conf`/`iou`/`imgsz` for watermark, `det`/`rec`/`min_score` for OCR), and `version` and `commit`: the source the image was built from.

Images decode every frame the same way (three-channel BGR, stored pixel order, no EXIF rotation), so boxes from different detectors compare.

## Calling it

Use `argus_cortex.moderation.NudeNetClient` (it parses `/infer` and reads `/health`):

```python
from argus_cortex.moderation import NudeNetClient

nn = NudeNetClient("http://<host>:8130")
nn.info().detector          # "640m"
nn.check(image_bytes)       # detections, labels, exposed_score
```

Settings, all optional: `NUDENET_RESOLUTION` (640), `ERAX_CONF` (0.2), `ERAX_IOU` (0.3), `WATERMARK_CONF` (0.2, a reporting floor below curator's own `min_conf`), `WATERMARK_IOU` (0.5), `WATERMARK_IMGSZ` (1024), `OCR_MIN_SCORE` (0.5: lines read with less confidence are dropped), `ARGUS_RETINA_MAX_PIXELS` (2**27: a larger image is a 422, refused from its header before it is decoded).

## Licence

AGPL-3.0. The servers build on AGPL libraries (NudeNet, Ultralytics). A service that only calls a retina server over HTTP is not affected. If you run a **modified** retina for other people, you must offer them its source. Every published image is built from a tagged commit of this repository, and its `/health` names that commit.

The model weights carry their own licences (table above).

## Develop

```bash
make install   # venv + editable install with the "dev" extras
make test      # the contract against a stand-in detector, the manifests and the pins
make lint
make images    # fetch the baked-in weights and build the images locally (IMAGES=nudenet for one)
```

Each model is a module (`argus_retina.nudenet`, `argus_retina.erax`, `argus_retina.watermark`, `argus_retina.ocr`) on the shared server in `argus_retina.sidecar`, with its Dockerfile, extra requirements and argus-iris `gpu-tenant.yaml` in `images/<model>/`. A release (`v*` tag) builds and pushes one image per model; nothing goes to PyPI.

Versions come from git tags (`hatch-vcs`). Scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template); run `copier update` to pull tooling changes, but keep the AGPL `LICENSE`, the licence fields in `pyproject.toml` and this repo's image workflows.
