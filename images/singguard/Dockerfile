# syntax=docker/dockerfile:1
# argus-retina SingGuard-4b (GPU: transformers, 4-bit bitsandbytes by default). Build from the repo root:
#   docker build -f images/singguard/Dockerfile -t argus-retina-singguard .
FROM python:3.12-slim

# gcc and libc6-dev: bitsandbytes imports its Triton kernels, and Triton compiles its CUDA driver stub on first use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY images/singguard/requirements.txt /tmp/requirements.txt
# The CUDA 12.4 wheels need a driver new enough for CUDA 12.4. torchvision comes from the same index, so it matches.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# No weights in the image: 8.9 GB would make every pull and CI build carry them. On its first load the server fetches
# them at a pinned Hub commit into SINGGUARD_DIR (mount a volume there) and checks each file's sha256 (pinned in
# src/argus_retina/singguard.py) before loading; later loads need no network.
ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina \
    && mkdir -p /models/singguard \
    && chown retina /models/singguard

USER retina
# The user has no home: Triton (under bitsandbytes) caches its kernels in $HOME, and huggingface_hub keeps its
# download locks in HF_HOME.
ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/tmp/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SINGGUARD_DIR=/models/singguard \
    SINGGUARD_QUANT=4bit \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-singguard" \
      org.opencontainers.image.description="SingGuard-4b behind argus-retina's /api/guard/check, /health and /admin routes (GPU; weights fetched to a volume)" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.revision="${COMMIT}"
VOLUME /models/singguard
EXPOSE 8080
# The first start downloads 8.9 GB before the model loads.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.singguard:app", "--host", "0.0.0.0", "--port", "8080"]
