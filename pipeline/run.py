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

from . import analyze, assign, audio, diarize, lang, roles, transcribe

STAGES = ["normalize", "transcribe", "diarize", "assign", "roles", "analyze"]


def run(audio_path: str, *, work_dir: str, language: str = "ml",
        model_size: str = "small", beam_size: int = 5,
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

    # --- 2. transcribe --------------------------------------------------
    # Check the language before spending 40 minutes transcribing in the wrong
    # one. Whisper does not fail on a wrong `language` - it returns fluent
    # nonsense, or an empty transcript, and everything downstream then
    # analyses that nonsense in perfect detail.
    progress("transcribe", 0.0, "checking language")
    detected, confidence, windows = transcribe.detect_language(wav)
    mismatch = detected != language and confidence >= 0.5
    if mismatch:
        progress("transcribe", 0.0,
                 f"WARNING sounds like '{detected}' not '{language}'")

    progress("transcribe", 0.0, f"whisper {model_size}, lang={language}")
    segments, meta = transcribe.transcribe(
        wav, language=language, model_size=model_size, beam_size=beam_size,
        progress=lambda done, total: progress("transcribe", done / (total or 1),
                                              f"{done:.0f}s / {total:.0f}s"),
    )
    # Did the model actually write the language, or English-looking noise?
    # A too-small model does not fail on Hindi - it invents plausible English.
    ratio = lang.script_ratio(" ".join(s["text"] for s in segments), language)
    meta.update({
        "script_ratio": ratio,
        "wrong_script": ratio is not None and ratio < 0.5,
        "requested_language": language,
        "detected_language": detected,
        "detected_confidence": confidence,
        "language_mismatch": mismatch,
        "language_windows": windows,
    })
    transcribe.save(segments, meta, os.path.join(work_dir, "segments.json"))
    progress("transcribe", 1.0, f"{len(segments)} segments")

    if not segments:
        hint = (f" The audio sounds like '{detected}' "
                f"(confidence {confidence}), but it was transcribed as "
                f"'{language}' - try again with the right language.") if mismatch else ""
        raise ValueError("no speech found in this audio." + hint)

    # --- 3. diarize -----------------------------------------------------
    progress("diarize", 0.0, "finding speakers")
    turns, dinfo = diarize.diarize(
        wav, num_speakers=num_speakers,
        min_speakers=min_speakers, max_speakers=max_speakers,
    )
    meta.update(dinfo)
    progress("diarize", 1.0, f"{dinfo['n_speakers']} speakers")

    # --- 4. assign ------------------------------------------------------
    progress("assign", 0.0, "matching words to voices")
    segments = assign.assign_speakers(segments, turns)
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
