"""
The whole pipeline, as one function over a file path.

    audio -> normalize -> transcribe -> diarize -> assign -> roles -> analyze

Every stage is a pure function over the stage before it, so each one can be
re-run alone against a saved result. That matters here more than usual: a
lesson takes minutes to transcribe, and you do not want to pay for it again
because the teacher-detection weights changed.
"""

import json
import os
import time

from . import analyze, assign, audio, backends, roles
from .transcribe import save as transcribe_mod_save

STAGES = ["normalize", "transcribe", "diarize", "assign", "roles", "analyze"]


def run(audio_path: str, *, work_dir: str, language: str = "ml",
        model_size: str = "small", beam_size: int = 5,
        backend: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None, max_speakers: int | None = None,
        use_llm: bool = True, llm_model: str = analyze.MODEL,
        on_progress=None) -> dict:
    """
    Returns the complete result dict. Also writes it to work_dir/result.json.

    on_progress(stage, fraction, note) is called throughout; the web app uses
    it to drive the status endpoint. Transcription and diarization each take
    minutes on CPU, so a job with no progress signal looks hung.
    """
    os.makedirs(work_dir, exist_ok=True)
    started = time.time()

    def progress(stage, fraction=0.0, note=""):
        if on_progress:
            on_progress(stage, max(0.0, min(fraction, 1.0)), note)

    # --- 1. normalize ---------------------------------------------------
    progress("normalize", 0.0, "decoding audio")
    wav, duration = audio.normalize(audio_path, os.path.join(work_dir, "audio_16k.wav"))
    progress("normalize", 1.0, f"{duration:.0f}s of audio")

    # --- 2/3/4. words, voices, and the marriage of the two ----------------
    # One call, two implementations: local faster-whisper + pyannote, or
    # Scribe doing both in a single request. Everything below is identical
    # either way - that is the whole point of the seam.
    engine = backends.get(backend)
    progress("transcribe", 0.0, f"backend: {engine.name}")
    segments, turns, meta = engine.run(
        audio_path=audio_path, wav_path=wav, language=language,
        progress=progress, model_size=model_size, beam_size=beam_size,
        num_speakers=num_speakers, max_speakers=max_speakers,
    )
    meta.setdefault("duration", round(duration, 2))
    transcribe_mod_save(segments, meta, os.path.join(work_dir, "segments.json"))

    if not segments:
        raise ValueError("no speech found in this audio")

    progress("assign", 0.0, "grouping into utterances")
    utterances = assign.to_utterances(segments)
    stats = assign.speaker_stats(utterances, duration)
    progress("assign", 1.0, f"{len(utterances)} utterances")

    # --- 5. roles -------------------------------------------------------
    progress("roles", 0.0, "identifying the teacher")
    verdict = roles.identify_teacher(utterances, stats, language, duration)
    utterances, names = roles.label_roles(utterances, verdict["teacher"])
    segments, _ = roles.label_roles(segments, verdict["teacher"])
    progress("roles", 1.0,
             f"{names.get(verdict['teacher'])} = {verdict['teacher']} "
             f"({verdict['confidence']} confidence)")

    # --- 6. analyze -----------------------------------------------------
    progress("analyze", 0.0, "analysing teacher speech")
    result_analysis = analyze.analyze(utterances, meta, language,
                                      use_llm=use_llm, model=llm_model)
    progress("analyze", 1.0, "done")

    result = {
        "meta": {**meta, "elapsed": round(time.time() - started, 1),
                 "source": os.path.basename(audio_path)},
        "teacher": verdict,
        "speakers": {sp: {**s, "label": names.get(sp, sp),
                          "role": "teacher" if sp == verdict["teacher"] else "student"}
                     for sp, s in stats.items()},
        "turns": turns,
        "segments": segments,
        "utterances": utterances,
        **result_analysis,
    }

    with open(os.path.join(work_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def reanalyze(work_dir: str, *, language: str = "ml", use_llm: bool = True,
              llm_model: str = analyze.MODEL, teacher: str | None = None) -> dict:
    """
    Redo everything after diarization from a saved result.json.

    Cheap: no model runs except the LLM. Use it to override a wrong teacher
    id, or to re-score after changing the role weights.
    """
    path = os.path.join(work_dir, "result.json")
    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    segments = assign.assign_speakers(result["segments"], result["turns"])
    utterances = assign.to_utterances(segments)
    stats = assign.speaker_stats(utterances, result["meta"].get("duration"))

    verdict = roles.identify_teacher(utterances, stats, language,
                                     result["meta"].get("duration"))
    if teacher:
        verdict = {**verdict, "teacher": teacher, "confidence": "manual",
                   "method": "override", "reasons": ["set by hand"]}

    utterances, names = roles.label_roles(utterances, verdict["teacher"])
    segments, _ = roles.label_roles(segments, verdict["teacher"])

    result.update({
        "teacher": verdict,
        "speakers": {sp: {**s, "label": names.get(sp, sp),
                          "role": "teacher" if sp == verdict["teacher"] else "student"}
                     for sp, s in stats.items()},
        "segments": segments,
        "utterances": utterances,
        **analyze.analyze(utterances, result["meta"], language,
                          use_llm=use_llm, model=llm_model),
    })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
