# Classroom Voice Analysis

Classroom audio goes in. A transcript, who spoke when, which voice is the
teacher, engagement metrics and a written summary come out.

```
audio -> normalize -> transcribe -> diarize -> assign -> roles -> analyze
         16k mono     words +       voices +   words to  who is   metrics +
                      timings       timings    voices    teaching AI review
```

Each stage is a pure function over the one before it. Transcription takes
minutes to hours, so nothing downstream is allowed to make you pay for it
twice: correcting a wrongly identified teacher and re-deriving every number
takes about a second, because it only touches the saved utterances.

## Two folders

```
assets/in/lesson.mp3  ->  python -m pipeline.batch  ->  assets/out/lesson/
                                                          result.json
                                                          transcript.txt
```

A session *is* a directory under `assets/out` holding a `result.json`. Nothing
else identifies one, so a finished analysis can be moved, copied or committed
as a plain folder.

Two things read that folder and neither writes to it. The local server lists it
on its home page, which is what you use while working. `pipeline.publish`
copies it into `site/`, which is the standalone demo you deploy.

## Running it

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in HF_TOKEN
```

`HF_TOKEN` is a free HuggingFace read token, and you have to **accept the model
terms once** at
[pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
with the same account, or it fails at load time. `ANTHROPIC_API_KEY` is
optional: without it every metric still runs and only the written summary is
skipped. No system `ffmpeg` is needed, audio is decoded through PyAV.

Check the token before committing an hour to a transcription:

```bash
.venv\Scripts\python -m pipeline.diarize
```

Then the loop:

```bash
# 1. put recordings in assets/in and process them
.venv\Scripts\python -m pipeline.batch --list     # what it would run, costs nothing
.venv\Scripts\python -m pipeline.batch

# 2. read them
.venv\Scripts\uvicorn web.app:app --port 8000     # localhost:8000

# 3. publish the library as a static site
.venv\Scripts\python -m pipeline.publish
```

`batch` skips anything that already has a result, so re-running it after adding
one file costs you one file. One failure does not stop the rest.

The server has two pages. `/` lists the library, and opening a session gives
you the full report plus a teacher override. `/process` uploads a single file
and streams it block by block, which is how you check a language you are unsure
about without waiting an hour to find out. An upload lands in the job store
rather than the library; a finished one gets a **Save to sessions** button.

`flow.md` has the full flag reference.

## Get the language right

This is the one setting that silently ruins a run.

Whisper does not error on a wrong `--lang`. It returns fluent nonsense, and
every number downstream is then computed faithfully over that nonsense. So the
pipeline samples five windows before transcribing and flags a mismatch, and it
checks `script_ratio`, the share of letters actually in the target script.

That script check catches *"this model is too small"*, because a model that
cannot write Devanagari falls back to English and the ratio collapses. It
cannot catch *"this is the wrong Devanagari language"*. Marathi transcribed as
Hindi comes back 98% Devanagari, passes every check, and means nothing.

The default is `hi`. The supplied recordings were made in Igatpuri,
Maharashtra, which is Marathi-speaking, so if a Hindi transcript reads oddly try
`--lang mr` on the same file before assuming the model is too small. Marathi
has a full lexicon here too, so nothing downstream is lost by switching.

Analysis on top of the transcript is lexical, and only some languages have a
lexicon: `hi`, `mr`, `ml`, `ta` and `en`. Everything else transcribes fine but
falls back to English cues only, so its question counts will be far too low
while the talk ratios still hold.

## Model size

Measured on 60s of the bundled recording, share of output actually in
Devanagari:

| Model | Devanagari |
|---|---|
| `tiny` | **0.0%** |
| `small` | **98.0%** |

`tiny` does not fail. It emits confident English that reads like a transcript
and is not one. `small` is the first usable model for Indic languages and is
the default for that reason. Use `tiny` to prove the pipeline runs, never to
produce a transcript you intend to read.

Adding a Devanagari `initial_prompt` to pin the script is the standard advice
and it measurably makes `small` worse (98.0% down to 86.5%). It is not used.

## Picking the teacher

pyannote returns `SPEAKER_00`, `SPEAKER_01` and no idea which is which. This is
the only genuinely inferential step, so it is scored from explicit features
rather than hidden in a model. Every verdict carries the reasons that produced
it, and `margin` says how close the runner-up was.

| Signal | Weight | Why |
|---|---|---|
| `talk_share` | 0.30 | teaching is asymmetric, the teacher holds the floor |
| `coverage` | 0.20 | the teacher is there start to finish, students come and go |
| `hub` | 0.15 | IRF structure puts the teacher on one side of most turn changes |
| `cue_rate` | 0.15 | "open your book" is said *to* a room |
| `question_rate` | 0.10 | the teacher asks, students mostly answer |
| `avg_turn` | 0.10 | explanation is long, answers are short |

A `low` confidence verdict means the top two scored within 0.08 of each other,
usually co-teaching, one dominant student, or a diarization that split one
person in two. The session page says so and lets you fix it, and the override
re-derives every downstream number without touching the audio.

## The metrics

Deterministic and offline, so they are comparable across sessions and never
drift. Each one carries its formula and how to read it, here and in the demo's
*How each metric is computed* panel.

| Metric | Formula | Reading it |
|---|---|---|
| **Teacher Dominance Ratio** | `teacher_talk_time / all speech time` | ~0.70 is typical. Sustained above 0.80 is lecture, not discussion. Below 0.50 usually means group work, or a mislabelled teacher. |
| **Student Participation** | per student: `their talk / all student talk`, plus `n_students_heard` | One student holding 80% of the student share is not participation. The headline "students talked 18%" hides that. The shares do not. |
| **Interaction Count** | consecutive `teacher -> student -> teacher` triads | High count with high dominance is brisk recitation. High count with low dominance is real dialogue. Near zero means students were spoken *at*. |
| **Question Response Rate** | `questions answered within 10s / teacher questions` | A low rate against many questions means they were rhetorical, or pitched past the class. Degrades exactly as far as the transcript does. |
| **Median Wait Time** | median gap from a teacher question to the next utterance | Rowe's finding: under ~3s students cannot formulate anything beyond recall. One of the few teaching changes with a large replicated effect. |
| **Silence Ratio** | `(duration - speech time) / duration` | Not a score. Deskwork and thinking are silence too. |

Talk ratios divide by speech, not wall clock, so silence is out of the
denominator. Speech the diarizer could not attribute to anyone stays in the
denominator and counts for nobody, because calling it student talk would
flatter the number.

The optional Claude review (`--llm`) is the other half: whether a question was
worth asking, whether an explanation landed, whether feedback was specific. It
is told the transcript is ASR output and to judge the teaching rather than the
transcription. Numbers without judgement flatter a lecture. Judgement without
numbers cannot be compared week to week.

## Swapping the transcription backend

Everything downstream is a pure function over one shape:

```python
segments: [{"start", "end", "text", "speaker"}]
turns:    [{"start", "end", "speaker"}]
```

So the whole transcribe-and-diarize half is swappable. `backends.py` holds two
implementations behind one call, chosen with `--backend` or `CVA_BACKEND`:
`local` is faster-whisper plus pyannote on your machine, `scribe` is ElevenLabs
Scribe v2 over the network.

`local` is the default because it works with no account, no key and no network,
and because for recordings of children "the audio never leaves the machine" is
a property worth defending. Scribe is faster and attributes every word to a
speaker directly, so there is no diarization to disagree with the transcript,
but the audio is uploaded and it costs per minute.

## Deploying

`site/` is plain HTML, CSS, JS and JSON with no build step. `vercel.json`
already configures Vercel (`framework: null`, no build command, output
directory `site`), so importing the repo needs no fields filled in and
`vercel --prod` works from a clean checkout. Cloudflare Pages is the same idea:
framework preset **None**, empty build command, build output directory `site`.

Two things catch people:

- **Run `pipeline.publish` before the commit that triggers the deploy**, or the
  deployed picker is empty. Static hosts serve what is committed.
- **Only the demo goes up.** Serverless platforms cap bundle size and request
  time. This is roughly 2GB of PyTorch running for tens of minutes, so it is not
  a size problem you can optimise away. The server stays local.

That split is the architecture, not a workaround. Classroom recordings are audio
of children taken in schools with unreliable connectivity, so a design where
inference is local and only derived JSON travels is the right shape anyway.
Hosting the server was tried: Render's free tier is 512MB and the pipeline is
~726MB resident before it opens a file.

If you do want the server elsewhere, `Dockerfile` and `docker-compose.yml` are
there. Give it dedicated vCPU rather than shared, or a GPU, and at least 4GB of
RAM. Set `CVA_PASSWORD`; compose refuses to start without it, deliberately,
because this holds recordings of children.

## Layout

```
pipeline/
  audio.py       any format -> 16k mono, via PyAV (no system ffmpeg)
  transcribe.py  faster-whisper, and the pre-flight language check
  diarize.py     pyannote 4.x, audio -> speaker turns, knows nothing of words
  assign.py      majority-overlap match of words to voices, with a confidence
  lang.py        question/cue/praise lexicons + script check
  roles.py       feature-scored teacher identification
  analyze.py     deterministic metrics + the Claude review
  run.py         the orchestrator, and reanalyze for cheap re-scoring
  stream.py      chunked orchestration: per-block output, then global speaker
                 reconciliation by voice embedding
  library.py     assets/in and assets/out: what exists, and is it safe to open
  batch.py       assets/in -> assets/out, resumable
  publish.py     the library -> site/data/, for the static demo
web/
  app.py         FastAPI: the library, upload, status, results, override
  jobs.py        SQLite-backed queue and a single worker
  static/        sessions, one session, process, live job
site/            the demo. Static, deployable as files, no server.
assets/
  in/            recordings waiting to be processed (gitignored, audio is big)
  out/           one directory per finished lesson. This is the library.
```

`site/report.js` and `site/app.css` are shared rather than copied: the server
mounts `site/` at `/site` and its own pages load them from there, so there is
one implementation of "render a result" and the live view cannot drift from the
published one.

Jobs run one at a time in a background worker. Whisper and pyannote each
saturate the CPU, so two lessons at once finish later than two in sequence.
State is in SQLite, so a restart does not lose the queue.

```bash
.venv\Scripts\python tests\test_pipeline.py     # 95 checks, no audio, no network
```

## Known limits

- **A wrong Devanagari language is invisible to the script check.** Marathi as
  Hindi passes at 98% and is still nonsense. The language detector is the only
  signal and it is weak on exactly that pair.
- **It is slow on CPU.** The bundled 64-minute lesson took 42 min at `tiny` and
  97 min at `small` on a 24-core box. A GPU is the only step change.
- **`--beam 1` is a trap on `small`.** It is much faster on `tiny`, but on
  `small` it measured slower and produced fewer segments: greedy decoding falls
  into repetition loops on noisy audio.
- **Thread count changes the output, not just the speed.** Identical audio gave
  48 segments at 16 threads and 10 at 24. Do not expect bit-reproducible
  transcripts across machines.
- **`n_students_heard` counts diarization clusters, not children.** Two quiet
  students at the back are often one cluster. The bundled lesson's metadata
  records 22 students; the diarizer hears 2 voices.
- **Segments straddling a speaker change** get the majority speaker and a
  confidence below 1.0, and the session page dims them. Splitting them properly
  needs word-level timestamps, which are not currently requested.
- **The lexicons are wordlists, not models.** Auditable, and wrong in
  predictable ways, which was the trade. Entries are three characters or more on
  purpose, because matching is substring containment and the bare Marathi
  interrogative "का" occurs inside ordinary words.
