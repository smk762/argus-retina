# syntax=docker/dockerfile:1
# argus-retina C2PA reader (CPU, no weights). Build from the repo root:
#   docker build -f images/c2pa/Dockerfile -t argus-retina-c2pa .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY images/c2pa/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina

USER retina
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES= \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-c2pa" \
      org.opencontainers.image.description="C2PA manifest reader behind POST /c2pa and /health: signed claims and the library's validation state" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND MIT" \
      org.opencontainers.image.revision="${COMMIT}"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.c2pa:app", "--host", "0.0.0.0", "--port", "8080"]
