# Classroom Voice Analysis

Classroom audio in, and back out: a transcript, who spoke when, which voice is
the teacher, engagement metrics, and a written summary of what the teacher
actually did.

```
audio ──▶ normalize ──▶ transcribe ──▶ diarize ──▶ assign ──▶ roles ──▶ analyze
          16k mono       words +        voices +    words to   who is    metrics +
                         timings        timings     voices     teaching  AI review
```

Each stage is a pure function over the one before it. That is the whole design:
transcription takes minutes to hours, so nothing downstream is allowed to make
you pay for it twice. Correcting a wrong teacher and re-deriving every number
takes about a second.

---

## Where the inference runs

**The demo is a static page. Nothing is transcribed in your browser, and there
is no inference server behind it.**

The pipeline runs on a machine that has the audio on it. It writes a JSON
document per lesson. Only that JSON is published, and the demo page reads it.

```
  a machine with the audio                         anywhere
 ┌──────────────────────────┐                   ┌──────────────┐
 │  mp3 ──▶ pipeline/ ──▶   │  result.json      │   site/      │
 │  whisper + pyannote      │ ────────────────▶ │   static     │
 │  (or ElevenLabs Scribe)  │  committed        │   page       │
 └──────────────────────────┘                   └──────────────┘
       minutes to hours                            instant
```

This is not a shortcut around a hosting problem, though it does happen to
avoid one. It is the shape the product wants. Classroom recordings are audio
of children, taken in schools with unreliable connectivity; the design that
survives that is one where audio never leaves the building and only derived
numbers travel. Inference is local and asynchronous, the viewer is a thin
reader of results, and the boundary between them is a JSON document you can
read.

The honest consequence, stated plainly rather than hidden behind an upload
button: **the static page cannot analyse a file you give it.** It shows the
lessons that were processed offline, named as they are named in the dataset.
To analyse something new, run the pipeline locally — which is the next
section, and takes one command.

---

## Running it locally

### Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in HF_TOKEN
```

No system `ffmpeg` is needed — audio is decoded through PyAV, which ships its
own. That holds for diarization too: pyannote 4.x would normally decode via
torchcodec, which *does* need FFmpeg's shared libraries and fails on Windows
with a wall of DLL errors. We hand it a preloaded waveform instead, so its
decoder is never reached.

Two credentials go in `.env`:

| Variable | Needed for | Without it |
|---|---|---|
| `HF_TOKEN` | speaker diarization | the pipeline stops at the diarize step |
| `ANTHROPIC_API_KEY` | the AI teaching summary | metrics still run; the summary is skipped |

`HF_TOKEN` is a free HuggingFace read token, and you must **accept the model
terms once** at
[pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
with the same account. A token without that acceptance fails at load time
rather than at request time.

Check it before committing an hour to a transcription:

```bash
.venv\Scripts\python -m pipeline.diarize
```

### The three commands

```bash
# 1. the local server: upload a recording, watch it process, read the result
.venv\Scripts\uvicorn web.app:app --port 8000          # then open localhost:8000

# 2. or the same pipeline from the CLI
.venv\Scripts\python main.py assets/audio.mp3 --lang mr --model small --out out

# 3. publish finished analyses into the static demo
.venv\Scripts\python -m pipeline.publish                # every finished job
.venv\Scripts\python -m pipeline.publish out            # or one --out directory
```

`publish` writes `site/data/<recording>.json` and rewrites
`site/data/manifest.json`, which is the list the demo's picker renders. Commit
`site/` and the demo is deployable as static files anywhere.

### Previewing the static demo

```bash
.venv\Scripts\python -m http.server -d site 8080        # then open localhost:8080
```

It has to be served rather than opened from disk: browsers block `fetch()` on
`file://`, so double-clicking `site/index.html` shows an explanatory box and no
data. Any static host works — GitHub Pages, Netlify, an S3 bucket, `nginx`.

The running server also mounts `site/` at `/site`, because the stylesheet and
the report renderer in there are the ones it serves to its own results page.
There is one implementation of "render a result", not two, so the live view
and the published view cannot drift apart.

### Tests

```bash
.venv\Scripts\python tests\test_pipeline.py
```

76 checks over every stage that is a pure function. No audio, no credentials,
no network.

---

## Get the language right

This is the one setting that will silently ruin a run, and it has two distinct
failure modes that look identical from the outside.

Whisper does not error on the wrong `--lang`. It returns *fluent nonsense* —
or an empty transcript — and every number downstream is then computed
faithfully over that nonsense. Forcing the language is still correct (letting
Whisper autodetect makes it flip mid-lesson on code-mixed speech), so instead
the pipeline samples five windows before transcribing, records what the audio
actually sounds like, and flags a mismatch in `meta.language_mismatch`, on the
results page, and in the error when a transcript comes back empty.

### The script check catches a small model, not a wrong language

`meta.script_ratio` is the share of letters in the script the language is
written in, and `meta.wrong_script` flags anything under 0.5. It reliably
catches *"this model is too small to write this language"*, because a model
that cannot write Devanagari falls back to English and the ratio collapses.

It cannot catch *"this is the wrong Devanagari language."* Hindi and Marathi
share a script, so transcribing Marathi audio as Hindi produces 98% Devanagari
and a script check that passes — and words that mean nothing. There is a
measured example of exactly that in this repository: the bundled recording
transcribed at `--lang hi --model small` returns confident Devanagari like
`लज्टें अई गल ख्या करव्ट` for over an hour.

So read the language detection, not just the script ratio:

| | detected | reading |
|---|---|---|
| `assets/audio.mp3`, 5 windows | `hi` ×4 (p = 0.54, 0.27, 0.95, 0.96), `ko` ×1 (p = 0.30) | mean **0.679** — weak, and one window was not even Indic |

A confident detection looks like a single language at p > 0.9 in every window.
That is not what this file does, which is a signal in itself: Whisper's
language ID does not distinguish Marathi from Hindi well, and the supplied
recordings come from **Igatpuri, Maharashtra** — Marathi-speaking. That is why
`pipeline/lang.py` sets `DEFAULT_LANGUAGE = "mr"`.

Treat that default as a guess about the dataset, not a fact about a file. Set
`--lang` per recording.

### Which languages are more than a transcript

Transcription works for anything Whisper supports. The *analysis* on top of it
is lexical, and only some languages have a lexicon:

| Language | Transcribes | Question / cue / praise lexicon |
|---|---|---|
| Marathi `mr` | yes | yes |
| Hindi `hi` | yes | yes |
| Malayalam `ml` | yes | yes |
| Tamil `ta` | yes | yes |
| English `en` | yes | yes |
| Telugu `te`, Kannada `kn`, everything else | yes | **no** — falls back to English cues only |

The English list is always included on top of the local one, because classroom
speech is code-mixed. For a language with no lexicon, that fallback is *all*
there is: question counts and cue scores will be far too low, so the talk
ratios and the timeline still hold but the question metrics do not.

## Model size is not a speed dial for Indian languages

Measured on 60s of the bundled recording, share of output actually in
Devanagari:

| Model | Config | Devanagari |
|---|---|---|
| `tiny` | as shipped | **0.0%** |
| `tiny` | + Devanagari `initial_prompt` | 11.4%, and garbage |
| `small` | **as shipped** | **98.0%** |
| `small` | + Devanagari `initial_prompt` | 86.5% — worse |
| `small` | + prompt, `condition_on_previous_text=False` | 93.8% — worse, 65% slower |

Two things follow.

**`tiny` cannot write Devanagari at all.** It does not fail; it emits confident
English ("Let's go to the last time") that reads like a transcript and is not
one. `small` is the first usable model for Indic languages, and is the default
for that reason. Use `tiny` to prove the pipeline runs, never to produce a
transcript you intend to read.

**Do not add an `initial_prompt` to pin the script.** It is the standard advice
and it measurably makes `small` worse. `small` already gets there on its own.

## How the teacher is identified

pyannote returns `SPEAKER_00`, `SPEAKER_01`… with no idea which is which.
Picking the teacher is the only genuinely inferential step here, so it is
scored from explicit features rather than hidden in a model — every verdict
carries the reasons that produced it, and `margin` says how close the runner-up
was.

| Signal | Weight | Why |
|---|---|---|
| `talk_share` | 0.30 | teaching is asymmetric; the teacher holds the floor |
| `coverage` | 0.20 | the teacher is there start to finish, students come and go |
| `hub` | 0.15 | IRF structure puts the teacher on one side of most turn changes |
| `cue_rate` | 0.15 | "open your book", "मुलांनो" — said *to* a room |
| `question_rate` | 0.10 | the teacher asks, students mostly answer |
| `avg_turn` | 0.10 | explanation is long, answers are short |

A `low` confidence verdict means the top two scored within 0.08 of each other —
usually co-teaching, one dominant student, or a diarization that split one
person across two labels. The results page says so and lets you override it;
`POST /api/jobs/{id}/teacher` re-derives everything from the saved result
without touching the audio.

## The metrics

Deterministic, offline, and comparable across sessions — no model in the loop,
so they never drift. Each carries its formula, its logic and how to read it,
both here and in the demo's *How each metric is computed* panel.

| Metric | Formula | Reading it |
|---|---|---|
| **Teacher Dominance Ratio** | `teacher_talk_time / all speech time` | ~0.70 is typical. Sustained above ~0.80 is lecture, not discussion. Below ~0.50 usually means group work — or a mislabelled teacher. |
| **Student Participation Indicator** | per student: `their talk_time / all student talk_time`, plus `n_students_heard` | One student holding 80% of the student share is not participation. The headline "students talked 18%" hides that; the shares do not. |
| **Interaction Count (IRF triads)** | consecutive `teacher → student → teacher` utterances | High count + high dominance is brisk recitation. High count + low dominance is real dialogue. Near zero means students were spoken *at*. |
| **Question Response Rate** | `questions answered within 10s / teacher questions` | A low rate against many questions means they were rhetorical, or pitched past the class. Degrades exactly as far as the transcript does. |
| **Median Wait Time** | median gap from a teacher question ending to the next utterance | Rowe's finding: under ~3s, students cannot formulate anything beyond recall. One of the few teaching changes with a large replicated effect. |
| **Silence Ratio** | `(duration − speech time) / duration` | Not a score. Deskwork and thinking are silence too; on a noisy recording it also absorbs whatever the diarizer declined to call speech. |

Talk ratios divide by *speech*, not wall clock, so silence is excluded from
their denominator. Speech the diarizer could not attribute to anyone stays in
the denominator and counts for nobody — calling it student talk would flatter
the number.

`review` is the other half: Claude reading the dialogue and judging what the
numbers can't — whether a question was worth asking, whether an explanation
landed, whether feedback was specific. It is told the transcript is ASR output
and to judge the teaching rather than the transcription. It is opt-in (`--llm`,
or the checkbox) because it is the only part that costs money.

Numbers without judgement flatter a lecture; judgement without numbers can't be
compared week to week. The report carries both.

## Two backends behind one seam

Everything downstream — roles, metrics, the review, the whole web app — is a
pure function over one shape:

```python
segments: [{"start", "end", "text", "speaker"}]
turns:    [{"start", "end", "speaker"}]
```

So the entire transcription-and-diarization half is swappable. `backends.py`
holds both implementations behind one call, chosen with `CVA_BACKEND` or
`--backend`:

| | `local` (default) | `scribe` |
|---|---|---|
| What | faster-whisper + pyannote | ElevenLabs Scribe v2 |
| Where | this machine | their servers |
| Cost | free | per audio minute |
| Speed | tens of minutes to hours on CPU | about the upload time |
| Indic accuracy | needs `small`; below that no Devanagari at all | vendor claims ≤10% WER Hindi, ≤5% Malayalam |
| Privacy | nothing leaves the box | audio is uploaded |
| Needs | `HF_TOKEN`, ~600MB of weights | `ELEVENLABS_API_KEY` |

`local` stays the default because it is the one that works with no account, no
key and no network — and because for recordings of children, "the audio never
leaves the machine" is a property worth defending.

### What the adapter has to reconcile

Scribe returns **words**, not segments: a flat list where each item is a
`word`, `spacing`, or `audio_event`, carrying optional `start`, `end` and
`speaker_id`. Getting to our shape means four decisions, all in
`words_to_segments`:

- **Only `word` items are speech.** `spacing` is punctuation and `audio_event`
  is laughter or a door. Counting either as a turn would skew every talk ratio.
- **Words with no timings are dropped** — `start`/`end` are optional in their
  schema, and a segment without a clock cannot be placed on a timeline.
- **Speaker ids are mapped in order of first appearance**, not parsed. The only
  guarantee worth relying on is that equal ids mean the same voice.
- **A pause longer than 0.8s splits a segment**, so one uninterrupted monologue
  does not become a single unreadable block.

There is one thing the hosted path gets for free: Scribe attributes every
*word* to a speaker, so there is no separate diarization to disagree with the
transcript. The whole class of "this segment straddles a speaker change" simply
does not arise, and `speaker_conf` is always 1.0.

Streaming is local-only. Scribe answers in one response, so there are no blocks
to emit as they finish; asking for both quietly runs the whole file through
Scribe instead of silently falling back to the local models.

## Streaming vs batch

By default a lesson is processed in ~5-minute blocks and each block's dialogue
is pushed to the browser as soon as it is done, so you read the transcript
while the rest is still running. `stream=false` on the upload runs the old
single-pass path instead.

The catch is speaker identity, which is *global*. pyannote clusters within
whatever audio you hand it, so block 3's `SPEAKER_00` has nothing to do with
block 7's. Streaming resolves that in four steps: diarize each block alone,
emit it under provisional ids, take a voice embedding per (block, speaker), and
once the audio runs out cluster those embeddings across the whole lesson to map
every local id onto a global one.

So early output is provisional by construction — two blocks can show what turns
out to be one person, and the labels change when reconciliation runs. The UI
says so. Nothing downstream is computed until after reconciliation.

Batch mode is still the more accurate of the two: diarizing 64 minutes at once
gives the clustering far more to work with than diarizing thirteen 5-minute
blocks and stitching them together.

## Stopping and deleting sessions

| Action | Where |
|---|---|
| Stop a running or queued session | **Stop** in the sessions table, or `POST /api/jobs/{id}/cancel` |
| Delete one session + its audio + results | **Delete** in the table, or `DELETE /api/jobs/{id}` |
| Delete every failed session | **Delete all failed sessions** button, or `DELETE /api/jobs?status=failed` |

Cancellation is **cooperative**, not instant. Whisper and pyannote are opaque
calls that cannot be interrupted partway, so a running job stops at the next
block boundary — up to one chunk later. A queued job that never started is
cancelled immediately. Whatever finished before the stop stays on disk.

This is another reason chunked mode is the default: in batch mode the only
checkpoints are between whole stages, so a stop can take much longer to land.

## API

| Endpoint | Does |
|---|---|
| `POST /api/upload` | file + options, returns a `job_id` |
| `GET /api/jobs` | queue with live progress |
| `GET /api/jobs/{id}` | one job's status, stage, progress |
| `GET /api/jobs/{id}/stream` | Server-Sent Events: each block's dialogue as it lands |
| `GET /api/jobs/{id}/result` | the full result document |
| `GET /api/jobs/{id}/transcript?role=teacher` | plain text, optionally one role |
| `POST /api/jobs/{id}/teacher` | override the teacher, re-derive downstream |
| `DELETE /api/jobs/{id}` | remove the job, its audio, and its results |
| `GET /api/health` | which credentials are present, disk free |

Jobs run one at a time in a background worker. Whisper and pyannote each
saturate the CPU, so two lessons at once finish later than two in sequence —
and a machine that looks hung is worse than a queue that looks slow. State is
in SQLite, so a restart doesn't lose the queue.

## Layout

```
pipeline/
  audio.py       any format -> 16k mono, via PyAV (no system ffmpeg)
  transcribe.py  faster-whisper; also the pre-flight language check
  diarize.py     pyannote 4.x; audio -> speaker turns, knows nothing of words
  assign.py      majority-overlap match of words to voices, with a confidence
  lang.py        question/cue/praise lexicons (mr, hi, ml, ta, en) + script check
  roles.py       feature-scored teacher identification
  analyze.py     deterministic metrics + the Claude review
  run.py         the orchestrator, and `reanalyze` for cheap re-scoring
  stream.py      the chunked orchestrator: per-block output, then global
                 speaker reconciliation by voice embedding
  publish.py     finished result.json -> site/data/, for the static demo
web/
  app.py         FastAPI: upload, status, results, override
  jobs.py        SQLite-backed queue and single worker
  static/        upload page, live results page
site/            THE DEMO. Static, deployable as files, no server.
  index.html     session picker + report
  report.js      the report renderer, shared with web/static/job.html
  app.css        the stylesheet, shared likewise
  data/          committed analyses + manifest.json
tests/
  test_pipeline.py   76 checks over every stage that is a pure function
```

`site/report.js` and `site/app.css` are shared, not copied: the server mounts
`site/` at `/site` and its results page loads them from there.

## Running it as a container

Only needed if you want the *server* somewhere other than your own machine —
the static demo needs no runtime at all.

```bash
cp .env.example .env          # set HF_TOKEN and CVA_PASSWORD at minimum
docker compose up -d --build
```

### The machine matters more than the platform

One lesson is tens of minutes of *sustained* CPU, so the usual cheap tiers are
the wrong shape: it is not request latency that hurts, it is throughput on a
single long job.

Extrapolated from a real 64-minute lesson measured end to end on a 24-core box
(42 min at `tiny`, 97 min at `small`). CPU work scales sublinearly with cores,
so these are estimates, not promises — but the ordering is reliable:

| Host | Cores | `tiny` | `small` |
|---|---|---|---|
| a 24-core workstation | 24 | 42 min | ~97 min |
| Hetzner CCX43 (dedicated) | 16 | ~56 min | ~130 min |
| Hetzner CCX33 / DO / Linode | 8 | ~91 min | ~210 min |
| a cheap 2-core cloud tier | 2 | ~4 hr | ~9 hr |
| **any modern NVIDIA GPU** | — | **minutes** | **minutes** |

The uncomfortable conclusion: **on CPU, every affordable host is slower than
the machine you already have.** Hosting the server buys availability, not
speed — which is a large part of why the demo is a static page and the
inference is not hosted at all.

If you do host it: use *dedicated* vCPU, not shared, or a GPU box (build with
`--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124`;
`transcribe.py` and `diarize.py` both detect CUDA on their own, and `CVA_DEVICE`
forces it either way). Not serverless — Vercel, Lambda, Cloud Run and friends
cap request time and lose local disk, and a job here runs for tens of minutes
and writes to a volume.

### Things that will bite you

- **Run one web worker.** The job queue is an in-process thread, so a second
  worker starts a second consumer of a queue it cannot see: two processes
  racing for the same SQLite rows, each saturating the CPU. Scale by giving the
  container more cores, never by adding workers. The Dockerfile pins
  `--workers 1`.
- **The volume is the product.** `/data` holds every transcript and the queue.
  Back it up; a container rebuild without it loses everything.
- **First run downloads ~600MB** of model weights into `/data/huggingface`.
  Slow once, then cached.
- **Set `CVA_PASSWORD`.** Compose refuses to start without it, deliberately —
  this holds recordings of children.
- **Give it RAM.** A 64-minute lesson is ~250MB of float32 held twice, plus the
  models — the pipeline measures ~726MB resident before any audio is processed,
  which is why a 512MB free tier cannot run it at all. 8GB is comfortable, 4GB
  is tight.

## Known limits

- **A wrong Devanagari language is invisible to the script check.** Marathi
  transcribed as Hindi passes at 98% script ratio and is still nonsense. The
  language detector is the only signal, and it is weak on exactly this pair.
- **Segments that straddle a speaker change** get the majority speaker and a
  `speaker_conf` below 1.0; the results page dims them. Splitting them properly
  needs word-level timestamps, which are not currently requested.
- **It is slow on CPU, and `small` is the reason.** Measured on a 24-core box
  with no GPU: `tiny` runs at ~3x realtime, `small` at ~0.5x. The bundled
  64-minute lesson took 42 min at `tiny` and 97 min at `small`, before
  diarization adds its own pass. Timings vary a lot run to run — treat them as
  order-of-magnitude. A GPU is the only step change.
- **`--beam 1` is a trap on `small`.** It is ~7x faster on `tiny`, but on
  `small` it measured *slower* and produced fewer segments: greedy decoding
  falls into repetition loops on noisy audio. Use it with `tiny` only.
- **Thread count changes the output, not just the speed.** Identical audio and
  settings gave 48 segments at 16 threads and 10 at 24 — different thread
  counts reorder float reductions and nudge the decoder. Do not expect
  bit-reproducible transcripts across machines.
- **`n_students_heard` counts diarization clusters, not children.** Two quiet
  students at the back are often one cluster; one student who moves seats can
  become two. The bundled lesson's own metadata records 22 students; the
  diarizer hears 2 voices.
- **The lexicons in `lang.py` are wordlists, not models.** They are auditable
  and wrong in predictable ways, which is the trade that was wanted. Entries
  are three characters or more on purpose — matching is substring containment,
  and the bare Marathi interrogative "का" occurs inside ordinary words.
