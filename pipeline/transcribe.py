"""
Audio -> segments.

The ONLY thing this module produces is a list of dicts:
    {"start": float, "end": float, "text": str, "speaker": None}

Every other part of the system is a pure function over that list.
Get this right and nothing downstream is hard.
"""

import json
import os
import subprocess
import tempfile

from faster_whisper import WhisperModel

# Cache the model at module level so it loads once, not per call.
_MODEL = None


def get_model(size: str = "small", compute_type: str = "int8"):
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(size, device="cpu", compute_type=compute_type)
    return _MODEL


def normalize(path: str) -> str:
    """Any audio -> 16kHz mono WAV. Everything downstream assumes this."""
    out = os.path.join(tempfile.gettempdir(), "classroom_16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def transcribe(wav_path: str, language: str = "ml", model_size: str = "small"):
    """
    Returns (segments, meta).

    language: force it. "ml" Malayalam, "hi" Hindi, "ta" Tamil, "en" English.
    Do NOT autodetect on code-mixed audio - it flips per segment and you
    get nonsense.
    """
    model = get_model(model_size)

    seg_iter, info = model.transcribe(
        wav_path,
        language=language,
        vad_filter=True,                                  # skips silence: faster + gives you silence gaps free
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
    )

    segments = [
        {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip(),
            "speaker": None,                              # filled in by the speaker step later
        }
        for s in seg_iter
        if s.text.strip()
    ]

    meta = {
        "language": info.language,
        "duration": round(info.duration, 2),
        "n_segments": len(segments),
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