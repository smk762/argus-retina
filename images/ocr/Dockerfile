# syntax=docker/dockerfile:1
# argus-retina OCR: RapidOCR with PP-OCRv6 small (ONNX, CPU). Build from the repo root:
#   docker build -f images/ocr/Dockerfile -t argus-retina-ocr .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY images/ocr/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# PaddleOCR's PP-OCRv6 small detector and multilingual recogniser, and the text-direction classifier (Apache-2.0),
# as RapidOCR's v3.10.0 ONNX exports. Each URL names that release and each checksum is the file's, so a re-upload or
# an error page fails the build instead of changing the reader. The server passes these paths to RapidOCR, which then
# never downloads a model at runtime.
ARG OCR_BASE=https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx
ARG OCR_DET_SHA256=090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f
ARG OCR_REC_SHA256=6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884
ARG OCR_CLS_SHA256=e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c
RUN mkdir -p /models/ocr && cd /models/ocr \
    && curl -fL --retry 3 -o PP-OCRv6_det_small.onnx "${OCR_BASE}/PP-OCRv6/det/PP-OCRv6_det_small.onnx" \
    && curl -fL --retry 3 -o PP-OCRv6_rec_small.onnx "${OCR_BASE}/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx" \
    && curl -fL --retry 3 -o ch_ppocr_mobile_v2.0_cls_mobile.onnx "${OCR_BASE}/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx" \
    && printf '%s  %s\n' \
        "${OCR_DET_SHA256}" PP-OCRv6_det_small.onnx \
        "${OCR_REC_SHA256}" PP-OCRv6_rec_small.onnx \
        "${OCR_CLS_SHA256}" ch_ppocr_mobile_v2.0_cls_mobile.onnx \
       | sha256sum -c -

ARG VERSION=0.0.0+docker
ARG COMMIT=
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home retina

USER retina
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES= \
    OCR_MIN_SCORE=0.5 \
    ARGUS_RETINA_COMMIT=${COMMIT}
LABEL org.opencontainers.image.title="argus-retina-ocr" \
      org.opencontainers.image.description="RapidOCR (PP-OCRv6 small) behind argus-retina's /infer and /health: text lines with boxes and scores" \
      org.opencontainers.image.source="https://github.com/smk762/argus-retina" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND Apache-2.0" \
      org.opencontainers.image.revision="${COMMIT}"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD ["python", "-m", "argus_retina.healthcheck"]
CMD ["python", "-m", "uvicorn", "argus_retina.ocr:app", "--host", "0.0.0.0", "--port", "8080"]
