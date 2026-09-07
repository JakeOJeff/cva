"""
Audio -> segments.

The ONLY thing this module produces is a list of dicts:
    {"start": float, "end": float, "text": str, "speaker": None}

Every other part of the system is a pure function over that list.
Get this right and nothing downstream is hard.
"""

import json

from faster_whisper import WhisperModel

# Cache models at module level so a size loads once, not per call. The web
# app processes jobs sequentially in one worker, so a plain dict is enough.
_MODELS: dict[tuple[str, str, int], WhisperModel] = {}

# faster-whisper's own default. Raising it was measured on a 24-core box and
# made no reliable difference - run-to-run variance was larger than the effect
# - so this stays at the library default. Note that changing it changes the
# *output*, not just the speed: different thread counts reorder float
# reductions and nudge the decoder down different paths.
CPU_THREADS = 0


def get_model(size: str = "small", compute_type: str = "int8",
              cpu_threads: int = CPU_THREADS) -> WhisperModel:
    key = (size, compute_type, cpu_threads)
    if key not in _MODELS:
        _MODELS[key] = WhisperModel(size, device="cpu", compute_type=compute_type,
                                    cpu_threads=cpu_threads)
    return _MODELS[key]


def detect_language(wav_path: str, model_size: str = "tiny", n_windows: int = 5):
    """
    What language does this recording actually sound like?

    Not used to *choose* the language - forcing it is still right, because
    per-segment autodetection flips around on code-mixed classroom speech.
    This exists so that forcing the wrong one is loud instead of silent:
    ask Whisper for the wrong language and it does not error, it returns
    fluent nonsense, or nothing at all.

    Samples several windows spread through the file, because the opening
    minute of a lesson is often shuffling and chatter.

    Returns (language, mean_probability, per_window).
    """
    from .audio import SAMPLE_RATE, decode

    model = get_model(model_size)
    samples = decode(wav_path)
    window = SAMPLE_RATE * 30

    if len(samples) <= window:
        starts = [0]
    else:
        usable = len(samples) - window
        starts = [int(usable * (i + 0.5) / n_windows) for i in range(n_windows)]

    votes: dict[str, list[float]] = {}
    per_window = []
    for start in starts:
        lang_code, prob, _ = model.detect_language(audio=samples[start:start + window])
        votes.setdefault(lang_code, []).append(prob)
        per_window.append({"at": round(start / SAMPLE_RATE, 1),
                           "language": lang_code, "probability": round(prob, 3)})

    # Most windows wins; ties break on mean confidence.
    best = max(votes, key=lambda k: (len(votes[k]), sum(votes[k]) / len(votes[k])))
    return best, round(sum(votes[best]) / len(votes[best]), 3), per_window


def _decode(model, wav_path, language, vad, progress, beam_size=5):
    kwargs = dict(language=language, beam_size=beam_size, vad_filter=vad)
    if vad:
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)

    seg_iter, info = model.transcribe(wav_path, **kwargs)

    segments = []
    for s in seg_iter:
        text = s.text.strip()
        if text:
            segments.append({
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": text,
                "speaker": None,                          # filled in by the speaker step later
            })
        if progress:
            progress(s.end, info.duration)
    return segments, info


def transcribe(wav_path: str, language: str = "ml", model_size: str = "small",
               progress=None, vad: bool = True, beam_size: int = 5,
               cpu_threads: int = CPU_THREADS):
    """
    Returns (segments, meta).

    language: force it. "ml" Malayalam, "hi" Hindi, "ta" Tamil, "en" English.
    Do NOT autodetect on code-mixed audio - it flips per segment and you
    get nonsense. Use detect_language() to check you forced the right one.

    progress: optional callback(seconds_done, total_seconds) - faster-whisper
    yields lazily, so this is the only honest way to report how far in we are.

    vad: Silero voice-activity filtering. On by default - it skips silence,
    which is usually faster and gives you the silence gaps for free. It is not
    reliable on every passage though: on a quiet stretch of a real classroom
    recording it scored everything as non-speech and returned an empty
    transcript, while a louder stretch of the same recording came back fine
    with 86% of the audio kept. The failure is silent - an empty transcript,
    not an error - so when VAD returns nothing at all we do not believe it,
    and redo the pass without it. `vad_fallback` in the meta records that.

    beam_size: 5 is the quality default. Dropping to 1 (greedy) is roughly 7x
    faster on `tiny`, but on `small` it measured *slower* on this audio and
    produced fewer segments - greedy decoding falls into repetition loops on
    noisy speech and burns steps on temperature-fallback retries. So treat it
    as a lever for the small models only, and check the output when you use it.
    """
    model = get_model(model_size, cpu_threads=cpu_threads)

    segments, info = _decode(model, wav_path, language, vad, progress, beam_size)

    vad_dropped_everything = vad and not segments
    if vad_dropped_everything:
        segments, info = _decode(model, wav_path, language, False, progress, beam_size)

    meta = {
        "language": info.language,
        "duration": round(info.duration, 2),
        "n_segments": len(segments),
        "model": model_size,
        "beam_size": beam_size,
        "vad": vad and not vad_dropped_everything,
        "vad_fallback": vad_dropped_everything,
    }
    return segments, meta


def save(segments, meta, path: str = "segments.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "segments": segments}, f,
                  ensure_ascii=False, indent=2)


def load(path: str = "segments.json"):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["segments"], d["meta"]
