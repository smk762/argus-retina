# syntax=docker/dockerfile:1
# argus-retina watermark detector (YOLO, CPU). Build from the repo root:
#   docker build -f images/watermark/Dockerfile -t argus-retina-watermark .
#
# No weights are baked in. Mount them at /models/watermark.pt. The suite runs fancyfeast's YOLO11x, which has no
# stated licence and so may not be redistributed in a public image (`make watermark-weights` fetches it, pinned).
# Without the file the server stays up, /health reports `error`, and the HEALTHCHECK fails.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY images/watermark/requirements.txt /tmp/requirements.txt
# CPU torch: this image must never take a GPU. torchvision comes from the same index, or ultralytics pulls a PyPI
# build that doesn't match torch and NMS fails at the first detect.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt

ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina \
    && mkdir -p /models

USER retina
# Ultralytics writes its settings under $HOME / YOLO_CONFIG_DIR; the user has no home.
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES= \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    WATERMARK_MODEL=/models/watermark.pt \
    WATERMARK_CONF=0.2 \
    WATERMARK_IOU=0.5 \
    WATERMARK_IMGSZ=1024 \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-watermark" \
      org.opencontainers.image.description="Watermark detector (YOLO, weights mounted at runtime) behind argus-retina's /infer and /health" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.revision="${COMMIT}"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.watermark:app", "--host", "0.0.0.0", "--port", "8080"]
