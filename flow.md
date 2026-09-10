# Everything else

`README.md` has the project structure. This is how to run it, what the numbers
mean, and what will bite you.

- [Setup](#setup)
- [Two folders](#two-folders)
- [The loop](#the-loop)
- [Batch](#batch)
- [Web app](#web-app)
- [One file from the command line](#one-file-from-the-command-line)
- [Publishing to the demo](#publishing-to-the-demo)
- [Running without AI](#running-without-ai)
- [Get the language right](#get-the-language-right)
- [Model size](#model-size)
- [Picking the teacher](#picking-the-teacher)
- [The metrics](#the-metrics)
- [Deploying](#deploying)
- [Tests](#tests)
- [Known limits](#known-limits)

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in HF_TOKEN
```

`HF_TOKEN` is a free HuggingFace read token, and you have to **accept the model
terms once** at
[pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
with the same account, or it fails at load time rather than at request time.
`ANTHROPIC_API_KEY` is optional: without it every metric still runs and only the
written summary is skipped.

No system `ffmpeg` is needed. Audio is decoded through PyAV, which ships its
own. That holds for diarization too: pyannote 4.x would normally decode via
torchcodec, which does need FFmpeg's shared libraries and fails on Windows with
a wall of DLL errors. It gets handed a preloaded waveform instead, so its
decoder is never reached.

Check the token before committing an hour to a transcription:

```bash
.venv\Scripts\python -m pipeline.diarize
```

## Two folders

```
assets/in/OD11163.mp3  ->  python -m pipeline.batch  ->  assets/out/OD11163/
                                                           result.json
                                                           transcript.txt
```

A session **is** a directory under `assets/out` holding a `result.json`. Nothing
else identifies one, so a finished analysis can be moved, copied or committed as
a plain folder and it stays a session.

Three things run here, and it is worth being clear about which is which:

| | What it is | Needs |
|---|---|---|
| `pipeline.batch` | the CLI. Everything in `assets/in` not done yet. | Python, `HF_TOKEN` |
| `web.app` | the local server. Home page lists `assets/out`, `/process` uploads one file and streams it. | same |
| `site/` | the demo. A static page over analyses already published. | nothing, it is files |

The demo computes nothing. Neither does the server's home page. Both read JSON
the pipeline produced.

## The loop

```bash
# 1. drop recordings in assets/in, then process them
.venv\Scripts\python -m pipeline.batch --list       # what it would run, costs nothing
.venv\Scripts\python -m pipeline.batch

# 2. read them
.venv\Scripts\uvicorn web.app:app --port 8000       # localhost:8000

# 3. publish the library as a standalone static site
.venv\Scripts\python -m pipeline.publish

# 4. preview that
.venv\Scripts\python -m http.server -d site 8080    # localhost:8080
```

Then commit `assets/out/` and `site/`, which are the deliverable. The audio in
`assets/in/` is gitignored: it is large, and it is not ours to republish. So are
the pipeline's own intermediates inside a session folder; only `result.json` and
`transcript.txt` are tracked.

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
    --llm                      also run the paid AI review (off by default)

A recording that already has a `result.json` is skipped, so this is safe to
re-run: add one file, run it again, and only the new one costs you an hour.
`--force` overrides that. One failure does not stop the rest, and the exit code
is non-zero if anything failed.

Settings apply to the whole batch, not per recording, so lessons in different
languages need separate runs. Because finished recordings are skipped, the
second call will not touch the first:

```bash
.venv\Scripts\python -m pipeline.batch --only OD11163 --lang mr
.venv\Scripts\python -m pipeline.batch --only OD11166 --lang hi
```

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

Set `CVA_PASSWORD` in `.env` and the whole app sits behind HTTP Basic auth, with
`CVA_USERNAME` defaulting to `teacher`. Leave it blank and the gate is off,
which is fine on localhost and wrong anywhere else: this holds recordings of
children.

Uploads go into the job store, not the library. A finished job gets a **Save to
sessions** button that copies it into `assets/out`. That is a button rather than
automatic because the job store is where the experiments live, a wrong language
or a model that turned out too small, and the library is what you chose to keep.

Jobs run one at a time in a background worker. Whisper and pyannote each
saturate the CPU, so two lessons at once finish later than two in sequence, and
a machine that looks hung is worse than a queue that looks slow. State is in
SQLite, so a restart does not lose the queue.

Cancellation is cooperative rather than instant. Whisper and pyannote are opaque
calls that cannot be interrupted partway, so a running job stops at the next
block boundary. Whatever finished before the stop stays on disk.

## One file from the command line

    .venv\Scripts\python main.py assets/in/lesson.mp3 --out out

Same pipeline, same flags as `batch`, but it writes wherever `--out` says
instead of into the library. Use it for a scratch run somewhere else.

Re-score an existing run without re-transcribing (seconds, not minutes):

    .venv\Scripts\python main.py x --out out --teacher SPEAKER_02

## Publishing to the demo

    .venv\Scripts\python -m pipeline.publish [targets...]

With no arguments it publishes every session in `assets/out`. Give it
directories or `result.json` paths to publish only those.

It writes one `site/data/<recording>.json` per session plus
`site/data/manifest.json`, which is the list the picker renders. The published
document drops `turns` and `segments`, pre-assignment intermediates the viewer
never reads and a third of the file size. Nothing else is changed, so a reviewer
can diff it against a local run.

Two runs of the same recording both survive; the second gets a `-2` suffix
rather than overwriting the first.

## Running without AI

The Claude summary is the only part that costs money. Turn it off and everything
else still runs, and nothing leaves your machine.

It is already off by default everywhere: add `--llm` to the CLI or tick the box
in the web app to turn it on. If `ANTHROPIC_API_KEY` is missing the review is
skipped with a note in the result rather than failing the run, so this is really
about not spending money when you do have a key configured.

You still get the full transcript with timestamps and speaker labels, who spoke
when and how much, which speaker is the teacher with the reasons and a
confidence, every talk and silence ratio, question counts, wait time, longest
monologue, IRF triads, and per-student participation shares. What you lose is
only the qualitative judgement.

Stay on the defaults while you are still tuning the language, the model size and
the speaker count. Once the transcript and the teacher call look right, do one
run with the review on.

## Get the language right

This is the one setting that silently ruins a run.

Whisper does not error on a wrong `--lang`. It returns fluent nonsense, and
every number downstream is then computed faithfully over that nonsense. So the
pipeline samples five windows before transcribing and flags a mismatch, and it
checks `script_ratio`, the share of letters actually in the target script.

That script check catches *"this model is too small"*, because a model that
cannot write Devanagari falls back to English and the ratio collapses. It cannot
catch *"this is the wrong Devanagari language"*. Marathi transcribed as Hindi
comes back 98% Devanagari, passes every check, and means nothing.

`assets/audio.mp3` detects as `hi` in four windows out of five, mean p = 0.68,
with one window voting Korean. A confident detection looks like one language
above p = 0.9 in every window, so that is a signal in itself.

The default is `hi`. The supplied recordings were made in Igatpuri,
Maharashtra, which is Marathi-speaking, so if a Hindi transcript reads oddly try
`--lang mr` on the same file before assuming the model is too small. Marathi has
a full lexicon here too, so nothing downstream is lost by switching.

The cheapest way to check is `/process` in the web app. It streams each block as
it finishes, so a wrong language is visible in a couple of minutes rather than
at the end of an hour.

Analysis on top of the transcript is lexical, and only some languages have a
lexicon: `hi`, `mr`, `ml`, `ta` and `en`. Everything else transcribes fine but
falls back to English cues only, so its question counts will be far too low
while the talk ratios still hold. The English list is always included on top of
the local one, because classroom speech is code-mixed.

## Model size

Measured on 60s of the bundled recording, share of output actually in
Devanagari:

| Model | Devanagari |
|---|---|
| `tiny` | **0.0%** |
| `small` | **98.0%** |

`tiny` does not fail. It emits confident English that reads like a transcript
and is not one. `small` is the first usable model for Indic languages and is the
default for that reason. Use `tiny` to prove the pipeline runs, never to produce
a transcript you intend to read.

Adding a Devanagari `initial_prompt` to pin the script is the standard advice
and it measurably makes `small` worse, 98.0% down to 86.5%. It is not used.

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
person in two. The session page says so and lets you fix it. The override
re-derives every downstream number from the saved utterances without touching
the audio, which takes about a second and is the whole point of keeping the
stages separate.

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
denominator and counts for nobody, because calling it student talk would flatter
the number.

The optional Claude review is the other half: whether a question was worth
asking, whether an explanation landed, whether feedback was specific. It is told
the transcript is ASR output and to judge the teaching rather than the
transcription. Numbers without judgement flatter a lecture. Judgement without
numbers cannot be compared week to week.

## Deploying

`site/` is plain HTML, CSS, JS and JSON with no build step. `vercel.json`
already configures Vercel (`framework: null`, no build command, output directory
`site`), so importing the repo needs no fields filled in and `vercel --prod`
works from a clean checkout. Cloudflare Pages is the same idea: framework preset
**None**, empty build command, build output directory `site`.

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
RAM. Set `CVA_PASSWORD`; compose refuses to start without it, deliberately.

## Tests

    .venv\Scripts\python tests\test_pipeline.py

95 checks over every stage that is a pure function. No audio, no credentials, no
network.

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
- **The lexicons are wordlists, not models.** Auditable, and wrong in predictable
  ways, which was the trade. Entries are three characters or more on purpose,
  because matching is substring containment and the bare Marathi interrogative
  "का" occurs inside ordinary words.
