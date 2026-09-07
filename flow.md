# Running it

## Running without AI (free, fully offline)

The Claude review is the only part that costs money. Turn it off and
everything else still runs — transcript, speaker separation, teacher
identification, and every metric. Nothing leaves your machine.

**Web app:** nothing — the AI review checkbox is off by default. Tick
"Also run the AI teaching review" only when you want it.

**CLI:** nothing — it is off by default. Add `--llm` to turn it on.

```
.venv\Scripts\python main.py assets/audio.mp3 --out out
```

You do **not** need an `ANTHROPIC_API_KEY` for this. If the key is missing the
review is skipped automatically with a note in the result rather than failing
the run, so this is really just about not spending money when you *do* have
a key configured.

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

Stay on the defaults while you are still tuning the language, the model size
and the speaker count. Once the transcript and the teacher call look right, do one run
with the review on. Correcting the teacher afterwards and re-deriving every
number costs nothing and takes about a second.

---

## Web app

    .venv\Scripts\uvicorn web.app:app --port 8000

Open http://localhost:8000, drop in a recording, **pick the language**, wait.
`http://localhost:8000/api/health` tells you which credentials it can see.

## CLI

    .venv\Scripts\python main.py assets/audio.mp3 --out out --model small

    --lang hi|ml|ta|te|kn|en   force the language (default hi; get this right)
    --model tiny|base|small|medium   (default tiny; small is hours on CPU)
    --beam 5                   beam width; 1 is faster on tiny, a trap on small
    --speakers 4               exact count, if you know it
    --max-speakers 6           ceiling, safer than an exact count (default 6)
    --llm                      also run the paid AI review (off by default)

Re-score an existing run without re-transcribing (seconds, not minutes):

    .venv\Scripts\python main.py x --out out --teacher SPEAKER_02

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
