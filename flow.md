# Running it

Three things run here, and it is worth being clear about which is which:

| | What it is | Needs |
|---|---|---|
| `web.app` | the local server. Upload a recording, watch it process, read the result, correct the teacher. | Python, `HF_TOKEN` |
| `main.py` | the same pipeline from the command line, one file at a time. | same |
| `site/` | the **demo**. A static page over analyses already computed. | nothing — it is files |

The demo computes nothing. It reads JSON that the first two produced. See
*Where the inference runs* in the README for why that is the intended shape
and not a shortcut.

---

## The whole loop

```bash
# 1. process a recording — server or CLI, they run the same pipeline
.venv\Scripts\uvicorn web.app:app --port 8000
#    ...or...
.venv\Scripts\python main.py assets/audio.mp3 --lang mr --model small --out out

# 2. publish what finished into the static demo
.venv\Scripts\python -m pipeline.publish            # every finished job
.venv\Scripts\python -m pipeline.publish out        # or just one --out dir

# 3. look at it
.venv\Scripts\python -m http.server -d site 8080    # localhost:8080
```

Then commit `site/` — that is the deployable artefact.

---

## Running without AI (free, fully offline)

The Claude summary is the only part that costs money. Turn it off and
everything else still runs — transcript, speaker separation, teacher
identification, and every metric. Nothing leaves your machine.

**Web app:** nothing to do — the AI review checkbox is off by default. Tick
"Also run the AI teaching review" only when you want it.

**CLI:** nothing to do — it is off by default. Add `--llm` to turn it on.

You do **not** need an `ANTHROPIC_API_KEY` for this. If the key is missing the
review is skipped automatically with a note in the result rather than failing
the run, so this is really just about not spending money when you *do* have a
key configured.

You still need `HF_TOKEN` — that one is free, see the README.

### What you still get without AI

- the full transcript, with timestamps and speaker labels
- who spoke when, and how much
- which speaker is the teacher, with the reasons and a confidence
- teacher vs student talk ratio, silence ratio
- question count, questions answered, median wait time
- longest monologue, IRF triads, per-student participation shares
- every formula behind them, in the demo's *How each metric is computed* panel

### What you lose

Only the qualitative judgement: whether a question was worth asking, whether
an explanation landed, whether feedback was specific.

### The recommended way to work

Stay on the defaults while you are still tuning the language, the model size
and the speaker count. Once the transcript and the teacher call look right, do
one run with the review on. Correcting the teacher afterwards and re-deriving
every number costs nothing and takes about a second.

---

## Web app

    .venv\Scripts\uvicorn web.app:app --port 8000

Open http://localhost:8000, drop in a recording, **pick the language**, wait.
`http://localhost:8000/api/health` tells you which credentials it can see.

A finished session's page ends with the command that publishes it to the
static demo.

## CLI

    .venv\Scripts\python main.py assets/audio.mp3 --out out

    --lang mr|hi|ml|ta|en      languages with a scoring lexicon (default mr)
    --lang te|kn|...           transcribes, but scores on English cues only
    --model tiny|base|small|medium   default small; below it, no Devanagari
    --beam 5                   beam width; 1 is faster on tiny, a trap on small
    --speakers 4               exact count, if you know it
    --max-speakers 6           ceiling, safer than an exact count (default 6)
    --backend local|scribe     default local; scribe needs ELEVENLABS_API_KEY
    --llm                      also run the paid AI review (off by default)

Re-score an existing run without re-transcribing (seconds, not minutes):

    .venv\Scripts\python main.py x --out out --teacher SPEAKER_02

Results land in `out/result.json` and `out/transcript.txt`.

## Publishing to the demo

    .venv\Scripts\python -m pipeline.publish [targets...]

With no arguments it publishes every finished job under `data/results/`. Give
it directories or `result.json` paths to publish only those.

It writes one `site/data/<recording>.json` per session plus
`site/data/manifest.json`. The published document drops `turns` and `segments`
— pre-assignment intermediates the viewer never reads, and a third of the file
size. Nothing else is changed, so a reviewer can diff it against a local run.

Two runs of the same recording (a re-run at a bigger model, say) both survive;
the second gets a `-2` suffix rather than overwriting the first.

## Get the language right

Whisper does not error on the wrong `--lang`. It returns fluent nonsense, or
nothing at all, and every number downstream is then computed faithfully over
that nonsense.

The pipeline samples five windows before transcribing and flags a mismatch, so
a *badly* wrong language is loud. A subtly wrong one is not: Hindi and Marathi
share the Devanagari script, so the script check passes at 98% while the words
mean nothing. `assets/audio.mp3` detects as `hi` in four windows out of five —
mean p = 0.68, with one window voting Korean — and the recordings come from
Igatpuri, Maharashtra, which is Marathi-speaking. Hence `--lang mr` as the
default. Verify it per recording; do not inherit it.

## Tests

    .venv\Scripts\python tests\test_pipeline.py

76 checks. No audio, no credentials, no network.
