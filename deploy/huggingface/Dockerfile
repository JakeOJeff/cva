# Hugging Face Spaces variant.
#
# Three things differ from the root Dockerfile, all forced by the platform:
#
#   UID 1000       Spaces runs the container as user 1000 and nothing else.
#                  The root image uses 10001 and chowns /data from a root
#                  entrypoint; here there is no root, so the user is created
#                  at that id and owns everything from the start.
#   Port 7860      the Spaces default. Must match app_port in README.md.
#   No /data       disk is wiped on every restart and /data does not exist at
#                  build time, so everything lives under the user's home and
#                  the ~600MB of weights re-download after a restart. That is
#                  acceptable for a demo and not for real use.
#
# Build this from the REPOSITORY ROOT:
#     docker build -f deploy/huggingface/Dockerfile .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is ctranslate2's OpenMP runtime; the wheel imports without it and
# then dies at the first transcribe() call.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    CVA_DATA_DIR=/home/user/data \
    # 2 shared vCPU: one hour of audio is an all-day job. Keep uploads short
    # enough that a visitor sees a result while they are still looking.
    CVA_MAX_AUDIO_MINUTES=5 \
    CVA_MAX_UPLOAD_MB=60

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --user --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY --chown=user pipeline/ ./pipeline/
COPY --chown=user web/ ./web/
COPY --chown=user main.py ./
COPY --chown=user tests/ ./tests/

RUN mkdir -p $HOME/data

EXPOSE 7860

# One worker: the job queue is an in-process thread, so a second worker would
# start a second consumer of a queue it cannot see.
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
