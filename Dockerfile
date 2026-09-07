# Classroom Voice Analysis
#
# Two things drive every choice in here.
#
# The CPU-only torch wheels. The default torch pulls ~2.5GB of CUDA libraries
# that are dead weight on a CPU host, and push the image past the size limit
# of several PaaS providers. The cpu index cuts the image to roughly a third.
# For a GPU host, build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
#
# Python 3.12 rather than 3.14. Everything here works on 3.14, but 3.12 has
# wheels for every dependency on every architecture, so a build never falls
# back to compiling ctranslate2 or scipy from source.

FROM python:3.12-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Models land on the mounted volume, so a restart does not re-download
    # ~600MB of Whisper and pyannote weights.
    HF_HOME=/data/huggingface \
    CVA_DATA_DIR=/data

WORKDIR /app

# libgomp is ctranslate2's OpenMP runtime; without it the wheel imports and
# then dies at the first transcribe() call.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --extra-index-url "${TORCH_INDEX}" -r requirements.txt

COPY pipeline/ ./pipeline/
COPY web/ ./web/
# The static demo. It is not optional here: web/app.py mounts it at /site and
# serves the stylesheet and report renderer from it, so a container without
# this directory fails at startup rather than merely losing the demo page.
COPY site/ ./site/
COPY main.py ./
COPY tests/ ./tests/

# Runs as a non-root user, but /data is a mounted volume whose ownership is
# set by the host, so chown it at start rather than at build time.
RUN useradd --create-home --uid 10001 cva
COPY --chown=cva:cva docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

# ONE worker, deliberately. The job queue is an in-process thread, so a second
# web worker would start a second consumer of a queue it cannot see - two
# processes racing for the same SQLite rows, each saturating the CPU. Scale by
# giving the container more cores, never by adding workers.
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
