# syntax=docker/dockerfile:1
# argus-retina faces (GPU: InsightFace buffalo_l + MiVOLO ONNX). Build from the repo root:
#   docker build -f images/faces/Dockerfile -t argus-retina-faces .
#
# No weights in the image. buffalo_l has no licence that allows publishing it, and MiVOLO's ONNX is produced by
# argus-curator mivolo-convert rather than downloaded. Mount the pack at /models/insightface and the ONNX at
# /models/faces/mivolo_v2_d1_384.onnx. Without them the server stays up, /health reports error, and the HEALTHCHECK fails.
FROM python:3.12-slim

WORKDIR /app
COPY images/faces/requirements.txt /tmp/requirements.txt
# insightface 0.7.3 publishes no wheel, so this builds the sdist (it needs g++, dropped again afterwards). It also
# depends on the CPU onnxruntime, which shares its import name with onnxruntime-gpu: drop whichever wheel landed,
# then install the GPU build so a session cannot run on the CPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 g++ \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && (pip uninstall -y onnxruntime || true) \
    && (pip uninstall -y onnxruntime-gpu || true) \
    && pip install --no-cache-dir onnxruntime-gpu==1.20.2 \
    && apt-get purge -y g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY images/faces/gpu-tenant.yaml /app/gpu-tenant.yaml
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina \
    && mkdir -p /models/insightface /models/faces \
    && chown -R retina /models

USER retina
ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    INSIGHTFACE_ROOT=/models/insightface \
    FACES_PACK=buffalo_l \
    MIVOLO_MODEL=/models/faces/mivolo_v2_d1_384.onnx \
    FACES_DET_THRESH=0.5 \
    FACES_GPU_MEM_LIMIT_MB=1024 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-faces" \
      org.opencontainers.image.description="InsightFace buffalo_l and MiVOLO behind POST /faces, /health and /admin (GPU; weights mounted)" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.revision="${COMMIT}"
VOLUME /models/insightface
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.faces:app", "--host", "0.0.0.0", "--port", "8080"]
