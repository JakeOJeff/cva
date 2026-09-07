# Running it

## Running without AI (free, fully offline)

The Claude review is the only part that costs money. Turn it off and
everything else still runs — transcript, speaker separation, teacher
identification, and every metric. Nothing leaves your machine.

**Web app:** untick **"Run the AI teaching review as well as the metrics"**
before you press the button.

**CLI:** add `--no-llm`.

```
.venv\Scripts\python main.py assets/audio.mp3 --lang hi --model small --out out --no-llm
```

You do **not** need an `ANTHROPIC_API_KEY` for this. If the key is missing the
review is skipped automatically with a note in the result rather than failing
the run, so `--no-llm` is really just about not spending money when you *do*
have a key configured.

You still need `HF_TOKEN` — that one is free, see the README.

### What you still get without AI

- the full transcript, with timestamps and speaker labels
- who spoke when, and how much
- which speaker is the teacher, with the reasons and a confidence
- teacher vs student talk ratio, silence ratio
- question count, questions answered, median wait time
- longest monologue, IRF triads, per-student participation shares

### What you lose

Only the qualitative judgement: whether a question was worth asking, whether
an explanation landed, whether feedback was specific.

### The recommended way to work

Run `--no-llm` while you are still tuning the language, the model size and the
speaker count. Once the transcript and the teacher call look right, do one run
with the review on. Correcting the teacher afterwards and re-deriving every
number costs nothing and takes about a second.

---

## Web app

    .venv\Scripts\uvicorn web.app:app --port 8000

Open http://localhost:8000, drop in a recording, **pick the language**, wait.
`http://localhost:8000/api/health` tells you which credentials it can see.

## CLI

    .venv\Scripts\python main.py assets/audio.mp3 --lang hi --model small --out out

    --lang ml|hi|ta|te|kn|en   force the language (get this right, see below)
    --model tiny|base|small|medium
    --speakers 4               exact count, if you know it
    --max-speakers 8           ceiling, safer than an exact count
    --no-llm                   metrics only, no API call

Re-score an existing run without re-transcribing (seconds, not minutes):

    .venv\Scripts\python main.py x --out out --teacher SPEAKER_02 --no-llm

Results land in `out/result.json` and `out/transcript.txt`.

## Get the language right

Whisper does not error on the wrong `--lang`. It returns fluent nonsense, or
nothing at all, and every number downstream is then computed faithfully over
that nonsense.

`assets/audio.mp3` detects as **Hindi at p=0.99** — use `--lang hi`. The
`--lang ml` this file used to carry produced **zero segments**. The pipeline
now samples the audio before transcribing and flags a mismatch, so a wrong
language is loud rather than silent.

## Tests

    .venv\Scripts\python tests\test_pipeline.py

33 checks. No audio, no credentials, no network.
