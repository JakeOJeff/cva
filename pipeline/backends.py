"""
Two ways to get from audio to speaker-attributed segments.

Everything downstream - roles, metrics, the review, the whole web app - is a
pure function over one shape:

    segments: [{"start", "end", "text", "speaker"}]
    turns:    [{"start", "end", "speaker"}]

So the entire transcription-and-diarization half is swappable, and both
implementations live here behind one call:

  local   faster-whisper + pyannote, on this machine. Nothing leaves the
          box, costs nothing to run, and is slow on CPU - tens of minutes
          to hours for a lesson.

  scribe  ElevenLabs Scribe v2, which transcribes and diarizes in a single
          request. Far more accurate on Indian languages than a local
          `small` (Hindi <=10% WER, Malayalam <=5%, against a local model
          that cannot write Devanagari at all below `small`), and returns in
          about the time it takes to upload. It costs money per minute and
          the audio leaves the machine.

Choose with CVA_BACKEND=local|scribe. `local` stays the default: it is the
one that works with no account, no key and no network.
"""

import os
import time

DEFAULT = os.environ.get("CVA_BACKEND", "local").strip().lower()


def available() -> dict[str, bool]:
    return {
        "local": True,
        "scribe": bool(os.environ.get("ELEVENLABS_API_KEY")),
    }


def get(name: str | None = None):
    name = (name or DEFAULT).strip().lower()
    if name == "scribe":
        return ScribeBackend()
    if name == "local":
        return LocalBackend()
    raise ValueError(f"unknown backend '{name}' - expected 'local' or 'scribe'")


# ----------------------------------------------------------------- local

class LocalBackend:
    """faster-whisper for the words, pyannote for the voices, assign to marry them."""

    name = "local"
    needs_network = False
    # faster-whisper and pyannote both read the normalised 16k wav.
    needs_wav = True

    def run(self, *, audio_path, wav_path, language, progress=None,
            model_size="tiny", beam_size=5, num_speakers=None,
            max_speakers=6, **_):
        from . import assign, diarize, lang, transcribe

        def step(stage, fraction, note=""):
            if progress:
                progress(stage, fraction, note)

        step("transcribe", 0.0, "checking language")
        detected, confidence, _ = transcribe.detect_language(wav_path)
        mismatch = detected != language and confidence >= 0.5
        if mismatch:
            step("transcribe", 0.0, f"WARNING sounds like '{detected}' not '{language}'")

        step("transcribe", 0.0, f"whisper {model_size}, lang={language}")
        segments, meta = transcribe.transcribe(
            wav_path, language=language, model_size=model_size, beam_size=beam_size,
            progress=lambda done, total: step("transcribe", done / (total or 1),
                                              f"{done:.0f}s / {total:.0f}s"))
        step("transcribe", 1.0, f"{len(segments)} segments")

        if not segments:
            hint = (f" The audio sounds like '{detected}' (confidence {confidence}), "
                    f"but it was transcribed as '{language}'.") if mismatch else ""
            raise ValueError("no speech found in this audio." + hint)

        step("diarize", 0.0, "finding speakers")
        turns, dinfo = diarize.diarize(wav_path, num_speakers=num_speakers,
                                       max_speakers=max_speakers)
        step("diarize", 1.0, f"{dinfo['n_speakers']} speakers")

        ratio = lang.script_ratio(" ".join(s["text"] for s in segments), language)
        meta.update(dinfo)
        meta.update({
            "backend": self.name,
            "script_ratio": ratio,
            "wrong_script": ratio is not None and ratio < 0.5,
            "requested_language": language,
            "detected_language": detected,
            "detected_confidence": confidence,
            "language_mismatch": mismatch,
        })
        return assign.assign_speakers(segments, turns), turns, meta


# ---------------------------------------------------------------- scribe

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SCRIBE_MODEL = os.environ.get("CVA_SCRIBE_MODEL", "scribe_v2")

# Scribe returns words, not segments. Consecutive words from one speaker are
# glued into a segment; a pause longer than this starts a new one, so a
# segment stays about the length of a sentence rather than a whole monologue.
WORD_GAP = 0.8


def _normalise_speaker(raw, order: dict) -> str:
    """
    Scribe's "speaker_0" -> our "SPEAKER_00".

    Ids are mapped in order of first appearance rather than parsed, because
    the only guarantee worth relying on is that equal ids mean the same
    voice - not that they are numbered from zero or numbered at all.
    """
    if raw is None:
        return "UNKNOWN"
    if raw not in order:
        order[raw] = f"SPEAKER_{len(order):02d}"
    return order[raw]


def words_to_segments(words, gap: float = WORD_GAP):
    """
    Scribe's word list -> our (segments, turns).

    Only `type == "word"` items carry meaning here. `spacing` is punctuation
    and whitespace, and `audio_event` is laughter or a door - neither is
    speech, and counting either as a speaker turn would skew every talk
    ratio downstream.
    """
    order: dict[str, str] = {}
    segments, turns = [], []
    cur = None

    for w in words:
        if w.get("type") != "word":
            continue
        start, end = w.get("start"), w.get("end")
        text = (w.get("text") or "").strip()
        if start is None or end is None or not text:
            continue

        speaker = _normalise_speaker(w.get("speaker_id"), order)
        if cur and cur["speaker"] == speaker and start - cur["end"] <= gap:
            cur["end"] = end
            cur["text"] = f"{cur['text']} {text}"
        else:
            if cur:
                segments.append(cur)
            cur = {"start": start, "end": end, "text": text, "speaker": speaker}

    if cur:
        segments.append(cur)

    # Turns are the same runs without the words. Adjacent segments from one
    # speaker merge, because a turn is "who held the floor", not "where the
    # sentence broke".
    for s in segments:
        if turns and turns[-1]["speaker"] == s["speaker"] \
                and s["start"] - turns[-1]["end"] <= gap:
            turns[-1]["end"] = s["end"]
        else:
            turns.append({"start": s["start"], "end": s["end"], "speaker": s["speaker"]})

    for s in segments:
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
        s["speaker_conf"] = 1.0        # Scribe attributes each word itself
        s["contested"] = False
    for t in turns:
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)

    return segments, turns


class ScribeBackend:
    """One POST does transcription and diarization together."""

    name = "scribe"
    needs_network = True
    # Scribe is handed the ORIGINAL file (see run() below), so decoding a
    # 16k wav here would cost a 115MB write and the memory to build it for
    # a waveform nothing ever opens. On a 512MB host that is the whole
    # difference between working and being OOM-killed at the first stage.
    needs_wav = False

    def run(self, *, audio_path, wav_path, language, progress=None,
            num_speakers=None, max_speakers=None, **_):
        import requests

        from . import lang

        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "CVA_BACKEND=scribe but ELEVENLABS_API_KEY is not set.\n"
                "  Create a key at https://elevenlabs.io/app/settings/api-keys\n"
                "  and put ELEVENLABS_API_KEY=... in your .env,\n"
                "  or set CVA_BACKEND=local to run everything on this machine.")

        def step(stage, fraction, note=""):
            if progress:
                progress(stage, fraction, note)

        # Send the ORIGINAL file, not the normalised wav: Scribe decodes
        # anything, and a 60MB mp3 uploads a lot faster than the 123MB wav
        # it expands to.
        source = audio_path if os.path.exists(audio_path) else wav_path
        size_mb = os.path.getsize(source) / 1e6
        step("transcribe", 0.05, f"uploading {size_mb:.0f}MB to Scribe")

        data = {
            "model_id": SCRIBE_MODEL,
            "diarize": "true",
            "timestamps_granularity": "word",
        }
        if language:
            data["language_code"] = language
        # Scribe takes an exact count, not a ceiling. Passing max_speakers as
        # num_speakers would force that many, so only an exact figure is sent.
        if num_speakers:
            data["num_speakers"] = str(num_speakers)

        started = time.time()
        with open(source, "rb") as f:
            response = requests.post(
                SCRIBE_URL,
                headers={"xi-api-key": key},
                data=data,
                files={"file": (os.path.basename(source), f)},
                timeout=(30, 3600),         # generous read: a long lesson takes a while
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Scribe returned {response.status_code}: {response.text[:400]}")

        payload = response.json()
        # Multi-channel recordings come back wrapped in `transcripts`.
        if "words" not in payload and payload.get("transcripts"):
            payload = payload["transcripts"][0]

        words = payload.get("words") or []
        if not words:
            raise ValueError("Scribe returned no words for this audio")

        step("diarize", 0.9, "mapping words to speakers")
        segments, turns = words_to_segments(words)
        if not segments:
            raise ValueError("Scribe returned words but none with timings")

        speakers = sorted({t["speaker"] for t in turns})
        detected = payload.get("language_code") or language
        confidence = round(float(payload.get("language_probability") or 0.0), 3)
        text = " ".join(s["text"] for s in segments)
        ratio = lang.script_ratio(text, language)

        meta = {
            "backend": self.name,
            "language": detected,
            "duration": round(max(t["end"] for t in turns), 2),
            "n_segments": len(segments),
            "model": SCRIBE_MODEL,
            "n_speakers": len(speakers),
            "speakers": speakers,
            "n_turns": len(turns),
            "diarization_model": SCRIBE_MODEL,
            "script_ratio": ratio,
            "wrong_script": ratio is not None and ratio < 0.5,
            "requested_language": language,
            "detected_language": detected,
            "detected_confidence": confidence,
            "language_mismatch": bool(language and detected
                                      and not detected.startswith(language)
                                      and confidence >= 0.5),
            "scribe_seconds": round(time.time() - started, 1),
            # Scribe attributes every word itself, so there is no separate
            # diarization to disagree with the transcript - the whole class
            # of "contested segment" simply does not arise.
            "vad": None,
            "vad_fallback": False,
        }
        step("diarize", 1.0, f"{len(speakers)} speakers, {len(segments)} segments")
        return segments, turns, meta
