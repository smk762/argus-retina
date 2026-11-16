# syntax=docker/dockerfile:1
# argus-retina EraX-NSFW V1.0 (YOLO11s, CPU). Build from the repo root:
#   docker build -f images/erax/Dockerfile -t argus-retina-erax .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY images/erax/requirements.txt /tmp/requirements.txt
# CPU torch: this image must never take a GPU. torchvision comes from the same index, or ultralytics pulls a PyPI
# build that doesn't match torch and NMS fails at the first detect.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# EraX-NSFW-V1.0 small (Apache-2.0). The URL names a Hub commit and the checksum is that file's, so a re-upload or an
# error page fails the build instead of changing the detector.
ARG ERAX_URL=https://huggingface.co/erax-ai/EraX-NSFW-V1.0/resolve/aea60ac8d2ebcbe0fcbb29e623eba99945b988a6/erax_nsfw_yolo11s.pt
ARG ERAX_SHA256=cd268d5ac84058fc9f3681bcc5446775e7ca1fdcf53948277a8b7ba12055eb10
RUN mkdir -p /models \
    && curl -fL --retry 3 -o /models/erax_nsfw_yolo11s.pt "${ERAX_URL}" \
    && echo "${ERAX_SHA256}  /models/erax_nsfw_yolo11s.pt" | sha256sum -c -

ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina

USER retina
# Ultralytics writes its settings under $HOME / YOLO_CONFIG_DIR; the user has no home.
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES= \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    ERAX_MODEL=/models/erax_nsfw_yolo11s.pt \
    ERAX_CONF=0.2 \
    ERAX_IOU=0.3 \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-erax" \
      org.opencontainers.image.description="EraX-NSFW V1.0 (YOLO11s) behind argus-retina's /infer and /health" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND Apache-2.0" \
      org.opencontainers.image.revision="${COMMIT}"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.erax:app", "--host", "0.0.0.0", "--port", "8080"]
