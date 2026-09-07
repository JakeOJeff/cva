# Render (free tier) variant - Scribe backend only.
#
# The root Dockerfile builds a machine that transcribes. This one builds a
# machine that *asks something else* to transcribe, which is what makes it fit
# in a free 512MB / 0.1-CPU container - the local pipeline measures 726MB of
# RSS at rest, so it is not a tight fit, it is 1.4x over the cap. See
# requirements-scribe.txt for the full measurement.
#
# Four things differ from the root Dockerfile:
#
#   requirements-scribe.txt   no torch, no pyannote, no faster-whisper.
#                             ~1.5GB image becomes ~150MB, 726MB becomes
#                             ~200MB.
#   No apt layer              libgomp existed for ctranslate2's OpenMP runtime,
#                             which left with faster-whisper.
#   ${PORT}                   Render assigns the port and routes to it. Shell
#                             form so the variable actually expands.
#   No /data volume           disks are a paid Render feature. Everything lives
#                             on the container's ephemeral disk and is LOST on
#                             every restart and deploy. Acceptable for a demo,
#                             wrong for real transcripts.
#
# Build from the REPOSITORY ROOT:
#     docker build -f deploy/render/Dockerfile .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 cva
USER cva
ENV HOME=/home/cva \
    PATH=/home/cva/.local/bin:$PATH \
    CVA_DATA_DIR=/home/cva/data \
    # The point of this image. Without it the app defaults to `local` and dies
    # at the first job on an import that was deliberately not installed.
    CVA_BACKEND=scribe \
    # The free ElevenLabs plan is about 30 minutes of audio per month, total.
    # A 5-minute ceiling makes that roughly six demos rather than three, and
    # stops one visitor spending the month in a single upload.
    CVA_MAX_AUDIO_MINUTES=5 \
    CVA_MAX_UPLOAD_MB=60

WORKDIR $HOME/app

COPY --chown=cva:cva requirements-scribe.txt .
RUN pip install --user -r requirements-scribe.txt

COPY --chown=cva:cva pipeline/ ./pipeline/
COPY --chown=cva:cva web/ ./web/
COPY --chown=cva:cva main.py ./

RUN mkdir -p $HOME/data

# One worker: the job queue is an in-process thread, so a second worker starts
# a second consumer of a queue it cannot see. This holds even here, where jobs
# are mostly spent waiting on someone else's API.
CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
