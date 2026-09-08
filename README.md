# Classroom Voice Analysis

Classroom audio goes in. A transcript, who spoke when, which voice is the
teacher, engagement metrics and a written summary come out.

```
audio -> normalize -> transcribe -> diarize -> assign -> roles -> analyze
         16k mono     words +       voices +   words to  who is   metrics +
                      timings       timings    voices    teaching AI review
```

Each stage is a pure function over the one before it. Transcription takes
minutes to hours, so nothing downstream is allowed to make you pay for it twice:
correcting a wrongly identified teacher and re-deriving every number takes about
a second, because it only touches the saved utterances.

**[flow.md](flow.md) is everything else:** how to run it, what each metric
means, the language setting that silently ruins a run, and the known limits.

## Project structure

```
assets/
  in/              recordings waiting to be processed
  out/             THE LIBRARY. One directory per finished lesson.

pipeline/          the analysis. No web framework anywhere in here.
  audio.py         any format -> 16k mono, via PyAV (no system ffmpeg)
  transcribe.py    faster-whisper, and the pre-flight language check
  diarize.py       pyannote 4.x, audio -> speaker turns, knows nothing of words
  assign.py        majority-overlap match of words to voices, with a confidence
  lang.py          question/cue/praise lexicons, and the script check
  roles.py         feature-scored teacher identification
  analyze.py       deterministic metrics, and the Claude review
  backends.py      one seam: local whisper+pyannote, or ElevenLabs Scribe
  run.py           the orchestrator, and reanalyze for cheap re-scoring
  stream.py        chunked orchestration: per-block output, then global
                   speaker reconciliation by voice embedding
  library.py       assets/in and assets/out: what exists, is it safe to open
  batch.py         assets/in -> assets/out, resumable
  publish.py       the library -> site/data/, for the static demo

web/               the local server. Thin: every decision lives in pipeline/.
  app.py           FastAPI: the library, upload, status, results, override
  jobs.py          SQLite-backed queue and a single worker
  static/
    sessions.html  home page, lists the library
    session.html   one session's report
    process.html   upload and options
    job.html       a running upload, streamed block by block

site/              THE DEMO. Static, deployable as files, no server.
  index.html       session picker plus report
  report.js        the report renderer
  app.css          the stylesheet
  data/            published analyses plus manifest.json

main.py            one file through the pipeline, from the command line
tests/             95 checks. No audio, no credentials, no network.
Dockerfile         the server in a container, if you want it elsewhere
vercel.json        deploys site/ as a static page
```

Three things run, and it is worth being clear about which is which.

**`pipeline.batch`** turns everything in `assets/in` into a directory under
`assets/out`. A session *is* that directory: a `result.json` and a
`transcript.txt`, nothing else, so a finished analysis can be moved, copied or
committed as a plain folder.

**`web.app`** is the local server. Its home page lists `assets/out`, and
`/process` uploads a single recording and streams it as it transcribes.

**`site/`** is the demo. It computes nothing, and neither does the server's home
page. Both read JSON the pipeline already produced, which is why the deployed
version needs no runtime at all.

`site/report.js` and `site/app.css` are shared rather than copied. The server
mounts `site/` at `/site` and its own pages load them from there, so there is
one implementation of "render a result" and the live view cannot drift from the
published one.
