# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image) and the image must
# have the git binary -- python:*-slim does not ship one, and without it the
# install below fails with "unable to detect version".
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN uv pip install --system --no-cache "."

CMD ["python", "-c", "import argus_retina; print(argus_retina.__version__)"]

