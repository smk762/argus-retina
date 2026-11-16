# syntax=docker/dockerfile:1
# argus-retina NudeNet 640m. Build from the repo root, after `make weights`:
#   docker build -f images/nudenet/Dockerfile -t argus-retina-nudenet .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY images/nudenet/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# NudeNet's own v3.4-weights 640m.onnx (AGPL-3.0, as NudeNet). GitHub serves release assets to an anonymous
# client as a login page, so `make weights` fetches it with `gh` into the build context, and the build checks it.
# (argus-cull fetched a third-party Hugging Face mirror: the same file with its ONNX IR version lowered from 10 to 9.)
ARG NUDENET_640M_SHA256=04fe3d77980780c1f8297dc6d7f942fd5b3abe6942a188f742a85241e4f634eb
COPY images/nudenet/weights/640m.onnx /models/640m.onnx
RUN echo "${NUDENET_640M_SHA256}  /models/640m.onnx" | sha256sum -c -

# hatch-vcs can't read git here (no .git in the context): the release passes the tag's version and commit.
ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina

USER retina
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES= \
    NUDENET_MODEL=/models/640m.onnx \
    NUDENET_RESOLUTION=640 \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-nudenet" \
      org.opencontainers.image.description="NudeNet 640m behind argus-retina's /infer and /health" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.revision="${COMMIT}"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.nudenet:app", "--host", "0.0.0.0", "--port", "8080"]
