# Running it

Two folders are the whole interface.

```
assets/in/OD11163.mp3   ──▶  python -m pipeline.batch  ──▶  assets/out/OD11163/
                                                              result.json
                                                              transcript.txt
```

A session **is** a directory under `assets/out` holding a `result.json`.
Nothing else identifies one, so a finished analysis can be moved, copied or
committed as a plain folder and it stays a session.

Three things run here, and it is worth being clear about which is which:

| | What it is | Needs |
|---|---|---|
| `pipeline.batch` | the CLI. Everything in `assets/in` that has not been done yet. | Python, `HF_TOKEN` |
| `web.app` | the local server. Home page lists `assets/out`; `/process` uploads one file and streams it. | same |
| `site/` | the **demo**. A static page over analyses already published. | nothing — it is files |

The demo computes nothing. Neither does the server's home page. Both read JSON
that the pipeline produced. See *Where the inference runs* in the README for
why that is the intended shape and not a shortcut.

---

## The whole loop

```bash
# 1. drop recordings in assets/in, then process them
.venv\Scripts\python -m pipeline.batch

# 2. read them
.venv\Scripts\uvicorn web.app:app --port 8000       # localhost:8000

# 3. publish the library as a standalone static site
.venv\Scripts\python -m pipeline.publish

# 4. preview that
.venv\Scripts\python -m http.server -d site 8080    # localhost:8080
```

Then commit `assets/out/` and `site/` — those are the deliverable. The audio in
`assets/in/` is gitignored: it is large, and it is not ours to republish.

---

## Batch

    .venv\Scripts\python -m pipeline.batch [options]

    --list                     show what would run, then stop
    --only NAME                only files whose name contains NAME (repeatable)
    --force                    re-run recordings that already have a result
    --lang hi|mr|ml|ta|en      languages with a scoring lexicon (default hi)
    --lang te|kn|...           transcribes, but scores on English cues only
    --model tiny|base|small|medium   default small; below it, no Devanagari
    --beam 5                   beam width; 1 is faster on tiny, a trap on small
    --speakers 4               exact count, if you know it
    --max-speakers 6           ceiling, safer than an exact count (default 6)
    --backend local|scribe     default local; scribe needs ELEVENLABS_API_KEY
    --llm                      also run the paid AI review (off by default)

A recording that already has a `result.json` is skipped, so this is safe to
re-run: add one file, run it again, and only the new one costs you an hour.
`--force` overrides that. One failure does not stop the rest, and the exit code
is non-zero if anything failed.

Start with `--list`. It costs nothing and tells you exactly what the hour is
going to be spent on.

## Web app

    .venv\Scripts\uvicorn web.app:app --port 8000

| | |
|---|---|
| `/` | **Sessions.** The library in `assets/out`, plus anything still waiting in `assets/in`. |
| `/sessions/<name>` | one session: transcript, metrics, summary, and the teacher override. |
| `/process` | upload one recording and watch it transcribe, block by block. |
| `/api/health` | which credentials it can see. |

Uploads go into the job store, not the library. A finished job gets a **Save to
sessions** button that copies it into `assets/out`. That is a button rather than
automatic because the job store is where the experiments live — a wrong
language, a model that turned out too small — and the library is what you chose
to keep.

Correcting the teacher on a session re-derives every downstream number from the
saved utterances. No audio is touched, so it takes about a second. That is the
whole point of keeping the stages separate.

## One file, from the command line

    .venv\Scripts\python main.py assets/in/lesson.mp3 --out out

Same pipeline, same flags as `batch`, writes wherever `--out` says instead of
into the library. Use it when you want a scratch run somewhere else.

Re-score an existing run without re-transcribing (seconds, not minutes):

    .venv\Scripts\python main.py x --out out --teacher SPEAKER_02

## Publishing to the demo

    .venv\Scripts\python -m pipeline.publish [targets...]

With no arguments it publishes every session in `assets/out`. Give it
directories or `result.json` paths to publish only those.

It writes one `site/data/<recording>.json` per session plus
`site/data/manifest.json`, which is the list the picker renders. The published
document drops `turns` and `segments` — pre-assignment intermediates the viewer
never reads, and a third of the file size. Nothing else is changed, so a
reviewer can diff it against a local run.

Two runs of the same recording (a re-run at a bigger model, say) both survive;
the second gets a `-2` suffix rather than overwriting the first.

---

## Running without AI (free, fully offline)

The Claude summary is the only part that costs money. Turn it off and
everything else still runs — transcript, speaker separation, teacher
identification, and every metric. Nothing leaves your machine.

**Batch and CLI:** nothing to do — it is off by default. Add `--llm` to turn it
on.

**Web app:** nothing to do — the AI review checkbox is off by default.

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
- every formula behind them, in the *How each metric is computed* panel

### What you lose

Only the qualitative judgement: whether a question was worth asking, whether
an explanation landed, whether feedback was specific.

### The recommended way to work

Stay on the defaults while you are still tuning the language, the model size
and the speaker count. Once the transcript and the teacher call look right, do
one run with the review on. Correcting the teacher afterwards and re-deriving
every number costs nothing and takes about a second.

---

## Get the language right

Whisper does not error on the wrong `--lang`. It returns fluent nonsense, or
nothing at all, and every number downstream is then computed faithfully over
that nonsense.

The pipeline samples five windows before transcribing and flags a mismatch, so
a *badly* wrong language is loud. A subtly wrong one is not: Hindi and Marathi
share the Devanagari script, so the script check passes at 98% while the words
mean nothing. `assets/audio.mp3` detects as `hi` in four windows out of five —
mean p = 0.68, with one window voting Korean.

The default is `hi`. Note that the recordings were made in Igatpuri,
Maharashtra, which is Marathi-speaking, so `--lang mr` is worth trying on any
recording whose Hindi transcript reads oddly — `mr` has a full lexicon here
too. Verify per recording; do not inherit the default.

The cheapest way to check is `/process` in the web app: it streams each block
as it finishes, so a wrong language is visible within a couple of minutes
rather than at the end of an hour.

## Tests

    .venv\Scripts\python tests\test_pipeline.py

95 checks. No audio, no credentials, no network.
