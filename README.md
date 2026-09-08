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

## Development

### Approach

Initially, a simple transcriber was made, featuring a CLI where you can process audio using python and faster whisper. Then, the diarization was done using pyAnnote and a final but additional bonus AI review using Anthropic SDK. There is a Hugging face model also used to compare and properly diarize and process the audio. A docker image was also built so that this can be pushed to the cloud to attain a web audio processor. If you are running this locally, you can either upload the lesson/classroom audio files into the assets/in folder, run the pipeline batch command ( which defaults to hindi and the small model), and get the required results in assets/out. The other method is that you can either run this locally or on a proper cloud cpu or container, where you can host it and directly drag and drop your audio files and get it processed on the web! That webapp is available in the /web folder and the current demo static site is served from the /site folder. The demo site features the classroom audios which I processed locally overnight using the batching method.

### Assumptions

I thought that a webapp where you can drag and drop audio to process them is what was required, which I had made and pushed, but it isn't served on the demo site because this requires proper CPU computing, and there were no free alternatives which could provide enough compute for it, because pyAnnote uses a lot of resources at runtime. This works when run locally or using a VPS which is a really great method, and also because this is mostly self hosted and doesn't use any backend dependancies. Then I just ran the audio through my pipeline locally and then served the results onto a static site where you can view the heuristics and all the data clearly without requiring a server.

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
