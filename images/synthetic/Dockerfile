# syntax=docker/dockerfile:1
# argus-retina synthetic (Community Forensics ViT-S/16 at 384, CPU). Build from the repo root:
#   docker build -f images/synthetic/Dockerfile -t argus-retina-synthetic .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY images/synthetic/requirements.txt /tmp/requirements.txt
# CPU torch: this image must never take a GPU. The assert fails the build if a later install swaps in a CUDA wheel.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch; assert torch.version.cuda is None, torch.version.cuda"

# Community Forensics ViT-S/16-384 (MIT). The URL names a Hub commit and the checksum is that file's, so a re-upload
# or an error page fails the build instead of changing the score. The same pin is in argus_retina.synthetic.
ARG SYNTHETIC_URL=https://huggingface.co/OwensLab/commfor-model-384/resolve/6076002bf0d9dd37537f965ee2f06f826c333b61/model.safetensors
ARG SYNTHETIC_SHA256=b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387
RUN mkdir -p /models/synthetic \
    && curl -fL --retry 3 -o /models/synthetic/model.safetensors "${SYNTHETIC_URL}" \
    && echo "${SYNTHETIC_SHA256}  /models/synthetic/model.safetensors" | sha256sum -c -

ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina

USER retina
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES= \
    HOME=/tmp \
    TORCH_HOME=/tmp/torch \
    HF_HUB_OFFLINE=1 \
    SYNTHETIC_MODEL=/models/synthetic/model.safetensors \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-synthetic" \
      org.opencontainers.image.description="Community Forensics ViT-S/16 behind POST /synthetic and /health: a generated-image probability" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND MIT" \
      org.opencontainers.image.revision="${COMMIT}"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.synthetic:app", "--host", "0.0.0.0", "--port", "8080"]
