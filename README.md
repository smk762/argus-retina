# argus-retina

The Argus suite's shared detector servers. Each server is a small, stateless model: an image goes in, and scores or detections come out. Each model ships as its own container image, and all of them speak one HTTP contract. Retina only senses. The calling stages own the thresholds and verdicts.

Part of the [Argus suite](https://github.com/smk762/argus-cortex/blob/dev/SUITE.md). Design: [argus-cortex#88](https://github.com/smk762/argus-cortex/issues/88). First job: [argus-cortex#79](https://github.com/smk762/argus-cortex/issues/79), one shared NudeNet 640m instead of a copy in every repo.

## Images

| Image | Model | Weights licence | Runs on |
|---|---|---|---|
| `ghcr.io/smk762/argus-retina-nudenet` | [NudeNet](https://github.com/notAI-tech/NudeNet) **640m** (v3.4 weights) at 640 px | AGPL-3.0 (NudeNet) | CPU (ONNX) |
| `ghcr.io/smk762/argus-retina-erax` | [EraX-NSFW V1.0](https://huggingface.co/erax-ai/EraX-NSFW-V1.0) small (YOLO11s) | Apache-2.0 | CPU (torch) |
| `ghcr.io/smk762/argus-retina-watermark` | Watermark detector, **no weights in the image**: mount fancyfeast's [YOLO11x](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection) | none stated, so never redistributed | CPU (torch) |

All listen on port 8080, run as a non-root user, never claim a GPU (`CUDA_VISIBLE_DEVICES` is empty), and are healthy once their model has loaded.

```bash
make watermark-weights        # the YOLO11x the watermark service mounts (pinned by Space commit and sha256)
docker compose up -d          # on 127.0.0.1:8130 (nudenet), :8131 (erax) and :8132 (watermark)
```

### Watermark: bring your own weights

The watermark image serves curator's `crops.detect` and ships without a model. Mount one at `/models/watermark.pt`; compose mounts `images/watermark/weights/watermark.pt`, or the file `WATERMARK_WEIGHTS` names. Without it the server stays up, and `/health` reports `error` with the reason.

The suite runs fancyfeast's YOLO11x, the detector behind JoyCaption's watermark tags. It states no licence, so no published image contains it. mnemic's CC-BY-4.0 YOLOv8s, which could have been baked in, was tried instead. On 32 watermarked images it caught 15 against the YOLO11x's 28 at curator's 0.4 cutoff, and its boxes rarely matched ([cortex#88](https://github.com/smk762/argus-cortex/issues/88#issuecomment-6913605897)). Any Ultralytics detector works if its classes suit the caller.

### NudeNet 640m, not 320n

The upstream `ghcr.io/notai-tech/nudenet` image and the `nudenet` wheel bundle only the 320n detector. 640m finds more, and the two score the same image differently, so **a threshold tuned on one does not carry over to the other**. Check your cutoffs when you move from 320n. `NudeNetClient.info()` reports which detector a server runs.

The 640m file is NudeNet's own release asset (sha256 `04fe3d77…34eb`). GitHub serves release assets to an anonymous client as a login page, so `make weights` fetches it with `gh`. argus-cull's sidecar used a third-party Hugging Face mirror of it: the same weights, with the ONNX IR version in the header lowered from 10 to 9.

## The contract

- `POST /infer`: multipart image fields `f1`, `f2`, …, answered by `{"prediction": [[{"class", "score", "box": [x, y, w, h]}, ...], ...], "success": true}`, one list per image in field order. This is NudeNet's shape; EraX answers in it with its own classes (`anus`, `make_love`, `nipple`, `penis`, `vagina`), and the watermark detector with `watermark`. `400` without images, `422` for a file that isn't an image, `503` while the model loads.
- `GET /health`: argus-cortex's `ResidentHealth` (`model_state`: `loading`, `cpu` or `error`; `busy`; `idle_s`), plus `detector`, `model` (the weights file) and the model's settings (`resolution` for NudeNet, `conf`/`iou` for EraX, `conf`/`iou`/`imgsz` for watermark), and `version` and `commit`: the source the image was built from.

Images decode every frame the same way (three-channel BGR, stored pixel order, no EXIF rotation), so boxes from different detectors compare.

## Calling it

Use `argus_cortex.moderation.NudeNetClient` (it parses `/infer` and reads `/health`):

```python
from argus_cortex.moderation import NudeNetClient

nn = NudeNetClient("http://<host>:8130")
nn.info().detector          # "640m"
nn.check(image_bytes)       # detections, labels, exposed_score
```

Settings, all optional: `NUDENET_RESOLUTION` (640), `ERAX_CONF` (0.2), `ERAX_IOU` (0.3), `WATERMARK_CONF` (0.2, a reporting floor below curator's own `min_conf`), `WATERMARK_IOU` (0.5), `WATERMARK_IMGSZ` (1024).

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

Each model is a module (`argus_retina.nudenet`, `argus_retina.erax`, `argus_retina.watermark`) on the shared server in `argus_retina.sidecar`, with its Dockerfile, extra requirements and argus-iris `gpu-tenant.yaml` in `images/<model>/`. A release (`v*` tag) builds and pushes one image per model; nothing goes to PyPI.

Versions come from git tags (`hatch-vcs`). Scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template); run `copier update` to pull tooling changes, but keep the AGPL `LICENSE`, the licence fields in `pyproject.toml` and this repo's image workflows.
