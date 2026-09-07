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
own.

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
- **Whisper accuracy on Indian-language classroom audio is mediocre** at
  `small`. `medium` is materially better and materially slower. Expect roughly
  real-time on CPU at `small` — a one-hour lesson takes about an hour.
- **`n_students_heard` counts diarization clusters, not children.** Two quiet
  students at the back are often one cluster; one student who moves seats can
  become two.
- The lexicons in `lang.py` are wordlists, not models. They are auditable and
  wrong in predictable ways, which is the trade that was wanted.
