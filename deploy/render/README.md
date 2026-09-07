# Render, on the free tier

A public URL that transcribes a clip in about the time it takes to upload it,
for no monthly cost. It gets there by not doing the expensive work: with
`CVA_BACKEND=scribe`, ElevenLabs transcribes and diarizes in one request and
this container is a thin front end over it.

This is an **alternative** deployment. `docker compose up` at the repository
root is unchanged and still runs the whole pipeline locally with
`CVA_BACKEND=local`, which is what to use if you would rather keep the audio
on your own machine.

## Why the local pipeline cannot go here

Render's free instance is 512MB of RAM and 0.1 CPU. Measured RSS of the local
pipeline, before a single second of audio is processed:

| | RSS |
|---|---|
| bare python | 19 MB |
| + fastapi / `web.app` | 50 MB |
| + `import torch` | 234 MB |
| + faster-whisper `tiny` (int8) | 339 MB |
| + pyannote diarization pipeline | **726 MB** |
| + 10 minutes of decoded audio | **726 MB** |

Two things follow. It is not a tight fit that could be tuned - it is 1.4x over
the cap at rest. And that last row is why **chunking the audio cannot help**:
adding ten minutes of waveform moved the number by nothing, because the cost
is three model runtimes resident in memory, not the audio passing through
them. Even deleting diarization entirely - the actual product - only reaches
339MB, and inference spikes from there.

Removing the models is the only lever. That is what this deployment does:
`requirements-scribe.txt` omits `torch`, `torchaudio`, `pyannote.audio` and
`faster-whisper`, about 1.4GB of wheels, and the container settles at ~200MB.

Nothing on the Scribe path imports them: `diarize.py` imports torch inside its
functions, and `transcribe.py` imports faster-whisper inside `get_model()`, so
the `save`/`load` pair that `run.py` calls on every job stays importable with
neither installed.

## What "free" means here

The ElevenLabs free plan is **10,000 credits per month, and speech-to-text
costs ~330 credits per minute** - so roughly **30 minutes of audio a month**,
total, across every visitor. At the 5-minute upload cap in the Dockerfile that
is about six demos.

Attach no card and it is genuinely free: when the credits run out, requests
fail until the month resets. Nothing bills you by surprise. If you outgrow it,
pay-as-you-go is ~$0.22 per hour of audio, so $5 buys about 22 hours - which
is worth comparing against the ~$25/month Render instance you would need to do
the same work on your own CPU, more slowly.

Free-plan output also requires attributing `elevenlabs.io`, and the free plan
grants no commercial rights. That language is written about generated voice
rather than transcripts and I would not rely on my reading of it; check the
terms if this stops being a portfolio demo.

## Deploy

1. **Get an ElevenLabs API key** at <https://elevenlabs.io/app/settings/api-keys>.
   No card needed for the free plan.

2. **Push this branch to GitHub.**

3. **Render → New → Blueprint**, point it at the repo. It reads `render.yaml`
   at the root and creates the service with the right Dockerfile and health
   check. (Or: New → Web Service → Docker, with Dockerfile Path
   `deploy/render/Dockerfile` and the context at the repo root.)

4. **Set the secrets** in the dashboard - `ELEVENLABS_API_KEY`, and
   `ANTHROPIC_API_KEY` if you want the AI review. Render generates
   `CVA_PASSWORD` for you; read it from the Environment tab, because you need
   it to log in. The username is `teacher`.

5. **Open the URL** and sign in as `teacher`.

## What you are giving up

- **Everything is deleted on restart.** Disks are a paid Render feature, so
  uploads, results and the SQLite queue live on the container's ephemeral
  disk. Redeploys, crashes and Render's own maintenance all wipe it.
- **It sleeps after ~15 minutes idle.** The first visit afterwards takes ~30
  seconds to wake. It does not interrupt a running job, because a Scribe job
  finishes in roughly the time the upload takes.
- **The audio leaves your machine.** It goes to ElevenLabs. That is a real
  change in posture for classroom recordings of children, and it is why
  `local` remains the default everywhere else.

## Checking it worked

Sign in, then open `/api/health`. To an anonymous caller it returns only
`{"ok": true}` - credentials and disk are operational detail it will not hand
out - but authenticated it tells you exactly what you need:

```json
{"backend_default": "scribe", "backends": {"local": true, "scribe": true}}
```

`backend_default` must be `scribe`. `backends.scribe` is `true` only when
`ELEVENLABS_API_KEY` is set. If `backend_default` says `local`, the
environment variable did not apply and the first job will fail on a
`ModuleNotFoundError` for `faster_whisper` - which is the intended, loud
failure, not a bug.

Note that `backends.local` always reports `true`. It describes the seam, not
this image: nothing checks whether the wheels are installed.
