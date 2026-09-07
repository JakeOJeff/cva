# Classroom Voice Analysis

Upload a lesson recording. Get back a transcript, who spoke when, which voice
is the teacher, and an analysis of what the teacher actually did.

```
audio ──▶ normalize ──▶ transcribe ──▶ diarize ──▶ assign ──▶ roles ──▶ analyze
          16k mono       words +        voices +    words to   who is    metrics +
                         timings        timings     voices     teaching  AI review
```

Each stage is a pure function over the one before it. That is the whole design:
transcription takes minutes, so nothing downstream is allowed to make you pay
for it twice. Correcting a wrong teacher and re-deriving every number takes
about a second.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in the tokens
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
| `ANTHROPIC_API_KEY` | the AI teaching review | metrics still run; the review is skipped |

`HF_TOKEN` is a free HuggingFace read token, and you must **accept the model
terms once** at
[pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
with the same account. A token without that acceptance fails at load time
rather than at request time.

Check it before committing an hour to a transcription:

```bash
.venv\Scripts\python -m pipeline.diarize
```

### "Access to model ... is restricted and you are not in the authorized list"

A 403 means the token authenticated but the account is not granted access.
Three causes, in the order they actually happen:

1. **The terms were never accepted.** Open the model page logged in and
   complete the access form. It has to say *"You have been granted access"*.
2. **The token belongs to a different account** than the one granted access.
3. **It is a fine-grained token.** These cannot read gated repos by default
   even when the account has access — enable *"Read access to contents of all
   public gated repos you can access"* on the token, or just use a plain
   **Read** token. This is the usual culprit.

If you have access to a different checkpoint instead, point at it without
touching code:

```
CVA_DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
```

## Running

```bash
.venv\Scripts\uvicorn web.app:app --port 8000     # then open localhost:8000
.venv\Scripts\python main.py assets/audio.mp3 --lang hi --model small
.venv\Scripts\python tests\test_pipeline.py
```

See `flow.md` for the full flag list.

## Model size is not a speed dial for Indian languages

Measured on 60s of the bundled Hindi recording, share of output actually in
Devanagari:

| Model | Config | Devanagari |
|---|---|---|
| `tiny` | as shipped | **0.0%** |
| `tiny` | + Devanagari `initial_prompt` | 11.4%, and garbage |
| `small` | **as shipped** | **98.0%** |
| `small` | + Devanagari `initial_prompt` | 86.5% — worse |
| `small` | + prompt, `condition_on_previous_text=False` | 93.8% — worse, 65% slower |

Two things follow.

**`tiny` cannot write Hindi at all.** It does not fail; it emits confident
English ("Let's go to the last time") that reads like a transcript and is not
one. `small` is the first usable model for Indic languages. Use `tiny` to
prove the pipeline runs, never to produce a transcript you intend to read.

**Do not add an `initial_prompt` to pin the script.** It is the standard
advice and it measurably makes `small` worse. `small` already gets there on
its own.

Because a wrong-script transcript looks fine until you read it, the pipeline
now measures it: `meta.script_ratio` is the share of letters in the script the
language is written in, and `meta.wrong_script` flags anything under 0.5. The
results page leads with a warning when it trips, and says which numbers still
mean something — the speaker timeline and talk ratios come from the audio, so
they survive; the words, question counts and any AI review do not.

## Get the language right

This is the one setting that will silently ruin a run.

Whisper does not error on the wrong `--lang`. It returns *fluent nonsense* —
or an empty transcript — and every number downstream is then computed
faithfully over that nonsense. Forcing the language is still correct (letting
Whisper autodetect makes it flip mid-lesson on code-mixed speech), so instead
the pipeline samples five windows before transcribing, records what the audio
actually sounds like, and flags a mismatch in `meta.language_mismatch`, on the
results page, and in the error when a transcript comes back empty.

The bundled `assets/audio.mp3` is a case in point: it detects as **Hindi at
p=0.99**, and the `--lang ml` in the original `flow.md` produced zero segments.

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
| `cue_rate` | 0.15 | "open your book", "മനസ്സിലായോ" — said *to* a room |
| `question_rate` | 0.10 | the teacher asks, students mostly answer |
| `avg_turn` | 0.10 | explanation is long, answers are short |

A `low` confidence verdict means the top two scored within 0.08 of each other —
usually co-teaching, one dominant student, or a diarization that split one
person across two labels. The results page says so and lets you override it;
`POST /api/jobs/{id}/teacher` re-derives everything from the saved result
without touching the audio.

## Hosting it

There is a Dockerfile, a compose file and a `/data` volume holding uploads,
results, the SQLite queue and the downloaded model weights.

```bash
cp .env.example .env          # set HF_TOKEN and CVA_PASSWORD at minimum
docker compose up -d --build
```

Then read the honest part before picking a host.

### The machine matters more than the platform

This is not a normal web app. One lesson is tens of minutes of *sustained*
CPU, so the usual cheap tiers are the wrong shape: it is not request latency
that hurts, it is throughput on a single long job.

Extrapolated from a real 64-minute lesson measured end to end on a 24-core box
(42 min at `tiny`). CPU work scales sublinearly with cores, so these are
estimates, not promises — but the ordering is reliable:

| Host | Cores | `tiny` | `small` |
|---|---|---|---|
| your 24-core machine | 24 | 42 min | ~80 min |
| Hetzner CCX43 (dedicated) | 16 | ~56 min | ~107 min |
| Hetzner CCX33 / DO / Linode | 8 | ~91 min | ~174 min |
| Hugging Face Spaces, free | 2 | ~4 hr | ~7.7 hr |
| **any modern NVIDIA GPU** | — | **minutes** | **minutes** |

The uncomfortable conclusion: **on CPU, every affordable host is slower than
the machine you already have.** Hosting buys availability, not speed.

### Choosing

**A CPU VPS, if overnight turnaround is fine.** Hetzner CCX33 (8 dedicated
vCPU, 32GB) is around €25/month and the best value here; DigitalOcean and
Linode equivalents cost more for the same cores. Use *dedicated* vCPU, not
shared — a shared instance gets throttled exactly when a lesson is running.
Deploy is `git clone`, `docker compose up -d`. Expect ~3 hours a lesson at
`small`.

**A GPU box, if turnaround matters.** This is the only thing that makes it
fast: hours become minutes. Build with the CUDA wheels and both stages move —

```bash
docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
```

`transcribe.py` and `diarize.py` both detect CUDA on their own; `CVA_DEVICE`
forces it either way. A dedicated GPU server (Hetzner GEX44, ~€184/month) is
predictable; on-demand pods (RunPod, Lambda) are ~$0.35/hour, which is far
cheaper if you process a few lessons a day and shut it down between.

**Not serverless.** Vercel, Lambda, Cloud Run and friends cap request time and
lose local disk. A job here runs for tens of minutes and writes to a volume.

### Things that will bite you

- **Run one web worker.** The job queue is an in-process thread, so a second
  worker starts a second consumer of a queue it cannot see: two processes
  racing for the same SQLite rows, each saturating the CPU. Scale by giving
  the container more cores, never by adding workers. The Dockerfile pins
  `--workers 1`.
- **The volume is the product.** `/data` holds every transcript and the queue.
  Back it up; a container rebuild without it loses everything.
- **First run downloads ~600MB** of model weights into `/data/huggingface`.
  Slow once, then cached.
- **Set `CVA_PASSWORD`.** Compose refuses to start without it, deliberately —
  this holds recordings of children.
- **Give it RAM.** A 64-minute lesson is ~250MB of float32 held twice, plus
  the models. 8GB is comfortable, 4GB is tight.

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

## Streaming vs batch

By default a lesson is processed in ~5-minute blocks and each block's dialogue
is pushed to the browser as soon as it is done, so you read the transcript
while the rest is still running. `stream=false` on the upload runs the old
single-pass path instead.

The catch is speaker identity, which is *global*. pyannote clusters within
whatever audio you hand it, so block 3's `SPEAKER_00` has nothing to do with
block 7's. Streaming resolves that in four steps: diarize each block alone,
emit it under provisional ids, take a voice embedding per (block, speaker),
and once the audio runs out cluster those embeddings across the whole lesson
to map every local id onto a global one.

So early output is provisional by construction — two blocks can show what
turns out to be one person, and the labels change when reconciliation runs.
The UI says so. Nothing downstream is computed until after reconciliation.

Batch mode is still the more accurate of the two: diarizing 64 minutes at once
gives the clustering far more to work with than diarizing thirteen 5-minute
blocks and stitching them together.

## What comes out

`metrics` are deterministic and comparable across sessions — teacher talk
ratio, question count and answer rate, median wait time, longest monologue,
IRF triads, per-student participation shares. These are the numbers you trend.

`review` is Claude reading the dialogue and judging what the numbers can't:
whether a question was worth asking, whether an explanation landed, whether
feedback was specific. It is told the transcript is ASR output and to judge the
teaching rather than the transcription.

Numbers without judgement flatter a lecture; judgement without numbers can't be
compared week to week. The report carries both.

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
  lang.py        multilingual question/cue/praise lexicons (ml, hi, ta, en)
  roles.py       feature-scored teacher identification
  analyze.py     deterministic metrics + the Claude review
  run.py         the orchestrator, and `reanalyze` for cheap re-scoring
  stream.py      the chunked orchestrator: per-block output, then global
                 speaker reconciliation by voice embedding
web/
  app.py         FastAPI: upload, status, results, override
  jobs.py        SQLite-backed queue and single worker
  static/        upload page, results page, one stylesheet
tests/
  test_pipeline.py   33 checks over every stage that is a pure function
```

## Known limits

- **Segments that straddle a speaker change** get the majority speaker and a
  `speaker_conf` below 1.0; the results page dims them. Splitting them properly
  needs word-level timestamps, which are not currently requested.
- **It is slow on CPU, and `small` is the reason.** Measured on a 24-core
  box with no GPU, transcribing 60s of the bundled recording: `tiny` runs at
  ~3x realtime, `small` at ~0.5x. So a 64-minute lesson is roughly 20 minutes
  on `tiny` and **1.5-3 hours** on `small`, before diarization adds its own
  pass. Timings vary a lot run to run — treat them as order-of-magnitude.
  A GPU is the only step change; everything else is a few percent.
- **`--beam 1` is a trap on `small`.** It is ~7x faster on `tiny`, but on
  `small` it measured *slower* and produced fewer segments: greedy decoding
  falls into repetition loops on noisy audio. Use it with `tiny` only.
- **Thread count changes the output, not just the speed.** Identical audio and
  settings gave 48 segments at 16 threads and 10 at 24 — different thread
  counts reorder float reductions and nudge the decoder. Do not expect
  bit-reproducible transcripts across machines.
- **`n_students_heard` counts diarization clusters, not children.** Two quiet
  students at the back are often one cluster; one student who moves seats can
  become two.
- The lexicons in `lang.py` are wordlists, not models. They are auditable and
  wrong in predictable ways, which is the trade that was wanted.
