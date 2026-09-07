# Running it

## Web app (the normal way)

    .venv\Scripts\uvicorn web.app:app --reload --port 8000

Open http://localhost:8000, drop in a recording, pick the language, wait.

## CLI (one file, verbose)

    .venv\Scripts\python main.py assets/audio.mp3 --lang hi --model small --out out

    --lang ml|hi|ta|te|kn|en   force the language (get this right, see below)
    --model tiny|base|small|medium
    --speakers 4               exact count, if you know it
    --max-speakers 8           ceiling, safer than an exact count
    --no-llm                   metrics only, no API call

Re-score an existing run without re-transcribing (seconds, not minutes):

    .venv\Scripts\python main.py x --out out --teacher SPEAKER_02 --no-llm

## Tests

    .venv\Scripts\python tests\test_pipeline.py
