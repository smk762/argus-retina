# argus-retina

The Argus suite's shared detector servers. Each server is a small, stateless model: an image (or text) goes in, and scores, detections, a policy verdict or a signed manifest come out. Each model ships as its own container image, and all of them speak one HTTP contract. Retina only senses. The calling stages own the thresholds and verdicts.

Part of the [Argus suite](https://github.com/smk762/argus-cortex/blob/dev/SUITE.md). Design: [argus-cortex#88](https://github.com/smk762/argus-cortex/issues/88). First job: [argus-cortex#79](https://github.com/smk762/argus-cortex/issues/79), one shared NudeNet 640m instead of a copy in every repo.

## Images

| Image | Model | Weights licence | Runs on |
|---|---|---|---|
| `ghcr.io/smk762/argus-retina-nudenet` | [NudeNet](https://github.com/notAI-tech/NudeNet) **640m** (v3.4 weights) at 640 px | AGPL-3.0 (NudeNet) | CPU (ONNX) |
| `ghcr.io/smk762/argus-retina-erax` | [EraX-NSFW V1.0](https://huggingface.co/erax-ai/EraX-NSFW-V1.0) small (YOLO11s) | Apache-2.0 | CPU (torch) |
| `ghcr.io/smk762/argus-retina-watermark` | Watermark detector, **no weights in the image**: mount fancyfeast's [YOLO11x](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection) | none stated, so never redistributed | CPU (torch) |
| `ghcr.io/smk762/argus-retina-ocr` | [RapidOCR](https://github.com/RapidAI/RapidOCR) with PaddleOCR's **PP-OCRv6 small** (multilingual Latin and CJK): text lines with boxes | Apache-2.0 | CPU (ONNX) |
| `ghcr.io/smk762/argus-retina-singguard` | [SingGuard-4b](https://huggingface.co/inclusionAI/SingGuard-4b) (on Qwen3-VL-4B-Instruct): a policy verdict on an image, text, or both. **Weights fetched on first start**, not in the image | Apache-2.0 | GPU (transformers, 4-bit) |
| `ghcr.io/smk762/argus-retina-faces` | [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (detection, ArcFace embedding, pose) and [MiVOLO v2](https://huggingface.co/iitolstykh/mivolo_v2) age. **Weights mounted**, not in the image | `buffalo_l` non-commercial; MiVOLO Apache-2.0 | GPU (ONNX) |
| `ghcr.io/smk762/argus-retina-synthetic` | [Community Forensics](https://huggingface.co/OwensLab/commfor-model-384) ViT-S/16 at 384: a probability the image was generated | MIT | CPU (torch) |
| `ghcr.io/smk762/argus-retina-c2pa` | C2PA manifest reader ([c2pa-python](https://github.com/contentauth/c2pa-python)): the signed claims and the library's validation state. No weights | MIT OR Apache-2.0 | CPU |

All listen on port 8080, run as a non-root user, and are healthy once their model has loaded. The CPU images never claim a GPU (`CUDA_VISIBLE_DEVICES` is empty). SingGuard and faces do.

```bash
make watermark-weights        # the YOLO11x the watermark service mounts (pinned by Space commit and sha256)
docker compose up -d          # on 127.0.0.1:8130 (nudenet), :8131 (erax), :8132 (watermark), :8133 (ocr), :8136 (synthetic) and :8137 (c2pa)
docker compose --profile gpu up -d   # SingGuard on :8134, faces on :8135
```

### SingGuard: one guard for the suite

SingGuard judges an image, a prompt, a model's reply, or an image with its prompt, against a moderation policy, and answers with a verdict and the rule that fired ([#9](https://github.com/smk762/argus-retina/issues/9)). It gives no scores. The policy text is the caller's (`argus_cortex.moderation.policies`); without one, SingGuard's own rules A to G apply. What to do with a verdict stays with the caller too.

`POST /api/guard/check` takes the JSON `argus_cortex.moderation.SingGuardClient` sends: `input` (text), `image_base64` (raw or a data URL), `policy`, `thinking`, and for a text-only check `input_role` (`user` or `assistant`). `conversation` (`[{role, content}]`) adds earlier turns, so a reply is judged in context. `thinking` is `fast` (a verdict and rule), `fast-slow` (SingGuard's own: a first-line verdict, then why), `slow` (why first, so the reasoning isn't anchored to a verdict made before it) or `cascade` (fast, then slow for what fast flagged; slow's verdict stands). The answer is `{safe, verdict, rule, explanation, categories, fast_verdict, model, thinking, elapsed_ms}`; a check with no readable verdict is a `500`, never a pass.

```python
from argus_cortex.moderation import SingGuardClient

guard = SingGuardClient("http://<host>:8134", thinking="cascade")
guard.check(image_bytes, text="the prompt that made it").safe
guard.check_text("a prompt, before it is run").rules
```

The weights (8.9 GB) are not in the image. On its first start the server fetches them from the Hub at a pinned commit into the `/models/singguard` volume, checks every file against the sha256 pinned in `argus_retina.singguard`, and refuses a file that doesn't match; later starts need no network. It loads in 4-bit (bitsandbytes, about 3.5 GB of VRAM); `SINGGUARD_QUANT=bf16` loads it in full precision (about 9 GB). Images are shrunk to 1536 px on the long side before the model sees them.

It holds VRAM, so argus-iris places it and can unload it: `POST /admin/unload` (and `/admin/evict`, the same here: a 4-bit model can't move to the CPU) and `/admin/wake`, behind `Authorization: Bearer $ARGUS_ADMIN_TOKEN` when that is set. While a check runs they answer `409`. After an unload the next check loads the model again.

It takes a GPU lease around its startup load and each check (`ARGUS_GPU_COORDINATOR=broker` asks argus-iris; `none`, the compose default, takes none). When the broker can't be reached, the lease falls back to the host lock file in `ARGUS_GPU_LOCK_DIR` (`/tmp/argus-gpu`), shared with the suite's other GPU services. **SingGuard runs as a non-root user, so it can only create that lock file if the directory is writable.** Docker creates a missing bind-mount directory as root with mode 755, so create it first, world-writable and sticky. Once the file exists, cortex makes it `0666`, so every user can lock it:

```bash
sudo install -d -m 1777 /tmp/argus-gpu
# survive reboots (/tmp is cleared):
echo 'd /tmp/argus-gpu 1777 root root -' | sudo tee /etc/tmpfiles.d/argus-gpu.conf
```

A refused lease is a `503` for that check, and the next check tries again.

### Faces: detection, embedding, age

`POST /faces` is phase 3 of [cortex#88](https://github.com/smk762/argus-cortex/issues/88) ([#10](https://github.com/smk762/argus-retina/issues/10)). It answers what InsightFace and MiVOLO see, and nothing about what to do with it: no clustering, no under-18 rule, no choice of which age wins. Those stay in curator. FairFace (race) and the person detector stay there too, for now.

Multipart fields `f1`, `f2`, … come back as `{"schema": "argus-retina.faces/v1", "faces": [[...], ...], "success": true}`, one list per image. A face is `bbox` (`[x, y, w, h]`), `det_score`, `embedding` (ArcFace, L2-normalised), `kps` (five keypoints), `pose` (`yaw`, `pitch`, `roll` in degrees), `landmark_3d_68`, `eye_open`, `mouth_open`, and MiVOLO's `age_years`, `gender` (`male` or `female`) and `gender_score`. A missing reading is `null`. `400` without images, `422` for a file that isn't one, `503` while the model loads or a GPU lease is refused.

The weights are not in the image. Mount InsightFace's `buffalo_l` pack at `/models/insightface` (so the ONNX files are in `models/buffalo_l/`) and the MiVOLO ONNX from `argus-curator mivolo-convert` at `/models/faces/mivolo_v2_d1_384.onnx`. Compose bind-mounts `images/faces/weights/insightface` and `images/faces/weights/mivolo_v2_d1_384.onnx`, or the paths `INSIGHTFACE_ROOT` and `MIVOLO_MODEL` name. `buffalo_l` is not redistributed. MiVOLO v2 is Apache-2.0; the file this image reads is the converted ONNX, not the Hub checkpoint.

It holds about 1 GB once loaded and about 1.5 GB at peak (measured on a 3090, curator's arena cap). Each CUDA session grows its arena only by what an allocation asks, up to `FACES_GPU_MEM_LIMIT_MB` (1024). ONNX Runtime does not return that arena on `/admin/unload`, so argus-iris should stop the container to free the card (`evict_by: stop`). The manifest asks to be dropped after 10 minutes idle and is not pinned, so a large ComfyUI or LoRA lease on the same card can be admitted. The lease kind is `faces`, interactive, around the startup load and each request. `/admin/wake` takes none.

### OCR: what the image says

The OCR image reads the text in an image so a caller can check what a caption claims it reads (argus-lens reconcile, [#5](https://github.com/smk762/argus-retina/issues/5)). It returns one detection per line of text, with the line in `text`, in reading order: rows top to bottom, then left to right. It reads rows, not columns, so two facing pages interleave. `score` is the recogniser's confidence; a line it can barely read (ornate script, Fraktur) has a low score rather than a cleaned-up guess. It reads what is printed, misspellings included. Words are sometimes run together (`afProfessor`), so a name check should compare without spaces.

It was chosen over docTR, EasyOCR and Florence-2 on 31 photos with hand-checked text (historical documents and plates, street signs). RapidOCR found 48 of 69 phrases at about 0.4 s per image on the CPU. Florence-2 large found 51 but took 8 s, and it wrote plausible text that isn't in the image ("Hosibolt Maria Espinosa" for Josep Maria Espinàs), which is what a text check exists to catch.

### Synthetic: a probability, not a label

`POST /synthetic` scores each image with Community Forensics, a ViT-S/16 trained on 2.7 million images from 4,803 generators (Park and Owens, CVPR 2025). Multipart fields `f1`, `f2`, … come back as `{"schema": "argus-retina.synthetic/v1", "scores": [{"synthetic": 0.91}, ...], "success": true}`. `synthetic` is the model's probability that the image was generated, from 0 to 1. The caller chooses the cutoff. This server does not answer `real` or `synthetic` as a label.

The weights are MIT, from `OwensLab/commfor-model-384` at Hub commit `6076002`, and they are baked in (sha256 `b89f3627…b387`). The eval preprocessing is the paper's: shortest side to 440, centre crop 384, ImageNet normalisation. It runs on the CPU.

A 2026 zero-shot study of open detectors put this one first, at about 75% mean accuracy, and also found that Flux, Firefly v4 and Midjourney v7 still fool most of them. It scores the whole frame. A partial edit, a screenshot, or a heavy recompress moves the number, and a diagram or a 3D render can score as generated without coming from one of those generators.

### C2PA: the signed manifest

`POST /c2pa` reads a [C2PA](https://c2pa.org/) manifest (Content Credentials) from the original bytes and checks it with `c2pa-python`. It returns the claims. It does not say whether the picture is real.

Multipart fields come back as `{"schema": "argus-retina.c2pa/v1", "reports": [{...}, ...], "success": true}`. A report is `present`, `validation_state`, `failures`, `claim_generator`, `digital_source_types`, `actions`, `ingredients`, `signature` (`issuer` and `time`) and `error`. No manifest is `present: false` with `error: null`: the file is unsigned. That is most files, photographs and generated images alike. A generator that signs its output puts a digital source type such as `trainedAlgorithmicMedia` on the `c2pa.created` action, and that string is what comes back. `validation_state` is the library's own word, using its default trust list. `Trusted` means the signature chains to an anchor on that list. `Valid` means the signature and the content hash check; an untrusted signer stays `Valid` and shows up in `failures` (for example `signingCredential.untrusted`). `Invalid` means the signature or the hash failed. `failures` lists the active manifest's failure codes. A container the library cannot read (a BMP, for one) is `present: false` with `error` set, so it is not reported as a clean unsigned file.

`400` without images, `422` for a file that is not one.

### Watermark: bring your own weights

The watermark image serves curator's `crops.detect` and ships without a model. Mount one at `/models/watermark.pt`; compose mounts `images/watermark/weights/watermark.pt`, or the file `WATERMARK_WEIGHTS` names. Without it the server stays up, and `/health` reports `error` with the reason.

The suite runs fancyfeast's YOLO11x, the detector behind JoyCaption's watermark tags. It states no licence, so no published image contains it. mnemic's CC-BY-4.0 YOLOv8s, which could have been baked in, was tried instead. On 32 watermarked images it caught 15 against the YOLO11x's 28 at curator's 0.4 cutoff, and its boxes rarely matched ([cortex#88](https://github.com/smk762/argus-cortex/issues/88#issuecomment-6913605897)). Any Ultralytics detector works if its classes suit the caller.

### NudeNet 640m, not 320n

The upstream `ghcr.io/notai-tech/nudenet` image and the `nudenet` wheel bundle only the 320n detector. 640m finds more, and the two score the same image differently, so **a threshold tuned on one does not carry over to the other**. Check your cutoffs when you move from 320n. `NudeNetClient.info()` reports which detector a server runs.

The 640m file is NudeNet's own release asset (sha256 `04fe3d77…34eb`). GitHub serves release assets to an anonymous client as a login page, so `make weights` fetches it with `gh`. argus-cull's sidecar used a third-party Hugging Face mirror of it: the same weights, with the ONNX IR version in the header lowered from 10 to 9.

## The contract

- `POST /infer`: multipart image fields `f1`, `f2`, …, answered by `{"prediction": [[{"class", "score", "box": [x, y, w, h]}, ...], ...], "success": true}`, one list per image in field order. This is NudeNet's shape; EraX answers in it with its own classes (`anus`, `make_love`, `nipple`, `penis`, `vagina`), and the watermark detector with `watermark`. The OCR image answers with `text`, and each of its detections adds the line it read as a `text` field. `400` without images, `422` for a file that isn't an image, `503` while the model loads.
- `GET /health`: argus-cortex's `ResidentHealth` (`model_state`: `loading`, `cpu`, `gpu`, `unloaded` or `error`; `busy`; `idle_s`; `gpu`, the memory figures, on SingGuard), plus `detector`, `device`, `model` (the weights file) and the model's settings (`resolution` for NudeNet, `conf`/`iou` for EraX, `conf`/`iou`/`imgsz` for watermark, `det`/`rec`/`min_score` for OCR, `revision`/`input` for the synthetic score, `revision`/`quant`/`force_thinking` for SingGuard), and `version` and `commit`: the source the image was built from.
- SingGuard serves `POST /api/guard/check` instead of `/infer`, and the admin routes (above). Faces serves `POST /faces` instead of `/infer`, and the same admin routes. The synthetic image serves `POST /synthetic`, and the C2PA image serves `POST /c2pa`. Neither returns a label.

Images decode every frame the same way (three-channel BGR, stored pixel order, no EXIF rotation), so boxes from different detectors compare.

## Calling it

Use `argus_cortex.moderation.NudeNetClient` (it parses `/infer` and reads `/health`):

```python
from argus_cortex.moderation import NudeNetClient

nn = NudeNetClient("http://<host>:8130")
nn.info().detector          # "640m"
nn.check(image_bytes)       # detections, labels, exposed_score
```

Settings, all optional: `NUDENET_RESOLUTION` (640), `ERAX_CONF` (0.2), `ERAX_IOU` (0.3), `WATERMARK_CONF` (0.2, a reporting floor below curator's own `min_conf`), `WATERMARK_IOU` (0.5), `WATERMARK_IMGSZ` (1024), `OCR_MIN_SCORE` (0.5: lines read with less confidence are dropped), `ARGUS_RETINA_MAX_PIXELS` (2**27: a larger image is a 422, refused from its header before it is decoded). SingGuard: `SINGGUARD_QUANT` (`4bit` or `bf16`), `SINGGUARD_MAX_NEW_TOKENS` (2048, the reasoning modes' budget), `SINGGUARD_MAX_NEW_TOKENS_FAST` (256), `SINGGUARD_FORCE_THINKING` (judge every check in one mode, whatever it asks for), `ARGUS_ADMIN_TOKEN`. Faces: `FACES_PACK` (`buffalo_l`), `FACES_DET_THRESH` (0.5), `FACES_GPU_MEM_LIMIT_MB` (1024, per CUDA session).

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

Each model is a module (`argus_retina.nudenet`, `argus_retina.erax`, `argus_retina.watermark`, `argus_retina.ocr`, `argus_retina.synthetic`, `argus_retina.c2pa`, `argus_retina.singguard`, whose request and parsing are in `argus_retina.guard`, `argus_retina.faces`) on the shared server in `argus_retina.sidecar`, with its Dockerfile, extra requirements and argus-iris `gpu-tenant.yaml` in `images/<model>/`. A release (`v*` tag) builds and pushes one image per model; nothing goes to PyPI.

Versions come from git tags (`hatch-vcs`). Scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template); run `copier update` to pull tooling changes, but keep the AGPL `LICENSE`, the licence fields in `pyproject.toml` and this repo's image workflows.
