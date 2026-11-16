# Changelog

All notable changes to this project are documented here. Versions come from git tags (`hatch-vcs`). A release's
section here becomes its GitHub Release notes, and the release fails before pushing images if the tag has none.

## [Unreleased]

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
- `compose.yaml` runs both on loopback (`:8130`, `:8131`). CI builds both images on every PR; a `v*` tag pushes
  them and creates the GitHub Release. Nothing goes to PyPI.
