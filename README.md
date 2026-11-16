# argus-retina

The Argus suite's shared detector servers. Each server is a small, stateless model: an image goes in, and scores or detections come out. Each model ships as its own container image, and all of them speak one HTTP contract. Retina only senses. The calling stages own the thresholds and verdicts.

Part of the [Argus suite](https://github.com/smk762/argus-cortex/blob/dev/SUITE.md). Design: [argus-cortex#88](https://github.com/smk762/argus-cortex/issues/88).

> **Status: scaffold.** No server has moved here yet. The first planned images are NudeNet (640m), EraX-NSFW and a watermark detector, moving from argus-cull's sidecars.

## Using a retina server

Call it over HTTP, for example with `argus_cortex.moderation.NudeNetClient`:

- `POST /infer`: multipart image fields `f1`, `f2`, …, answered by `{"prediction": [[{"class", "score", "box"}, ...]], "success": true}`.
- `GET /health`: the model's state and which model and resolution it runs.

## Licence

AGPL-3.0. The servers build on AGPL libraries (NudeNet, Ultralytics). A service that only calls a retina server over HTTP is not affected. If you run a **modified** retina for other people, you must offer them its source.

The model weights carry their own licences, which are listed per image.

## Develop

```bash
make install   # venv + editable install with the "dev" extras
make test
make lint
```

Versions come from git tags (`hatch-vcs`). Scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template); run `copier update` to pull tooling changes, but keep the AGPL `LICENSE` and the licence fields in `pyproject.toml`.
