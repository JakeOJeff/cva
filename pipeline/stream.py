"""
The pipeline, run in chunks, emitting results as it goes.

`run.py` processes a lesson as one unit: you wait an hour and then everything
appears. This module trades some accuracy for visibility - it works through
the audio in blocks and emits each block's dialogue as soon as that block is
done.

The hard part is that speaker identity is GLOBAL and chunking is LOCAL.
pyannote clusters within whatever audio it is given, so chunk 3's
"SPEAKER_00" has nothing to do with chunk 7's "SPEAKER_00". Resolving that is
the whole design here:

  1. each chunk is diarized on its own, and its speakers are emitted with
     provisional ids (c3_SPEAKER_00) so the UI can show them immediately
  2. a voice embedding is taken for every (chunk, local speaker) pair
  3. once the audio runs out, those embeddings are clustered across the whole
     lesson, which maps every local id onto a global one
  4. everything is relabelled, and only then do roles and metrics run

So early output is provisional by construction: a speaker shown as two people
in chunks 3 and 7 becomes one person at the end. The UI has to say so, and
nothing downstream may be computed from the provisional labels.
"""

import json
import os
import time

import numpy as np

from . import analyze, assign, audio, diarize, roles, transcribe

CHUNK_SECONDS = 300.0        # 5 minutes: long enough for clustering to have something to work with
BOUNDARY_SEARCH = 10.0       # look this far either side of a cut for a quiet moment
MIN_EMBED_SECONDS = 1.5      # below this there is not enough voice to identify anyone
EMBED_SECONDS = 20.0         # cap: more than this adds nothing and costs time
MATCH_THRESHOLD = 0.70       # cosine distance; above it, two voices are different people


# --------------------------------------------------------------- chunking

def _quietest_near(samples: np.ndarray, target: int, search: int) -> int:
    """
    Nudge a chunk boundary to the quietest point nearby.

    Cutting mid-syllable costs a word in both chunks and can invent a speaker
    turn out of half a phoneme. A cheap RMS scan finds somewhere better.
    """
    lo = max(0, target - search)
    hi = min(len(samples), target + search)
    if hi - lo < audio.SAMPLE_RATE:
        return target

    win = audio.SAMPLE_RATE // 10                      # 100ms resolution
    region = samples[lo:hi]
    n = len(region) // win
    if n < 2:
        return target
    energy = np.sqrt((region[:n * win].reshape(n, win) ** 2).mean(axis=1))
    return lo + int(energy.argmin()) * win


def plan_chunks(samples: np.ndarray, chunk_seconds: float = CHUNK_SECONDS):
    """Returns [(start_sample, end_sample), ...] cut at quiet points."""
    sr = audio.SAMPLE_RATE
    step = int(chunk_seconds * sr)
    search = int(BOUNDARY_SEARCH * sr)

    bounds, pos = [0], step
    while pos < len(samples):
        cut = _quietest_near(samples, pos, search)
        if cut <= bounds[-1] + sr:                     # never emit an empty chunk
            cut = min(pos, len(samples))
        bounds.append(cut)
        pos = cut + step
    bounds.append(len(samples))

    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a >= sr:                                # skip slivers
            out.append((a, b))
    return out or [(0, len(samples))]


# ------------------------------------------------------------- embeddings

def _speaker_embeddings(pipeline, chunk: np.ndarray, turns) -> dict[str, np.ndarray]:
    """
    One voice embedding per local speaker in this chunk.

    Their speech is concatenated (up to EMBED_SECONDS) and passed through the
    same embedding model pyannote uses internally, so the vectors live in the
    space its own clustering was tuned for.
    """
    import torch

    sr = audio.SAMPLE_RATE
    by_speaker: dict[str, list[np.ndarray]] = {}
    for t in turns:
        a, b = int(t["start"] * sr), int(t["end"] * sr)
        if b > a:
            by_speaker.setdefault(t["speaker"], []).append(chunk[a:b])

    names, waves = [], []
    for speaker, pieces in by_speaker.items():
        voice = np.concatenate(pieces)[: int(EMBED_SECONDS * sr)]
        if len(voice) < MIN_EMBED_SECONDS * sr:
            continue                                   # too little audio to identify
        names.append(speaker)
        waves.append(voice)

    if not names:
        return {}

    width = max(len(w) for w in waves)
    batch = np.zeros((len(waves), 1, width), dtype=np.float32)
    for i, w in enumerate(waves):
        batch[i, 0, : len(w)] = w

    with torch.no_grad():
        vectors = pipeline._embedding(torch.from_numpy(batch))
    return {name: np.asarray(v, dtype=np.float64) for name, v in zip(names, vectors)}


def reconcile(embeddings: dict[str, np.ndarray], max_speakers: int | None = None,
              num_speakers: int | None = None, threshold: float = MATCH_THRESHOLD):
    """
    local id -> global id, by clustering every chunk's voices together.

    This is the step that turns "chunk 3 speaker 0" and "chunk 7 speaker 1"
    into one person. Everything emitted before it ran is provisional.
    """
    keys = list(embeddings)
    if not keys:
        return {}
    if len(keys) == 1:
        return {keys[0]: "SPEAKER_00"}

    from sklearn.cluster import AgglomerativeClustering

    matrix = np.vstack([embeddings[k] for k in keys])
    # Vectors carrying NaN come from near-silent speech; they cannot be
    # clustered, so give each its own identity rather than poison the matrix.
    bad = ~np.isfinite(matrix).all(axis=1)
    good_idx = np.where(~bad)[0]

    labels = {}
    if len(good_idx) >= 2:
        if num_speakers:
            model = AgglomerativeClustering(n_clusters=min(num_speakers, len(good_idx)),
                                            metric="cosine", linkage="average")
        else:
            model = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                            linkage="average",
                                            distance_threshold=threshold)
        assigned = model.fit_predict(matrix[good_idx])

        if max_speakers and len(set(assigned)) > max_speakers:
            model = AgglomerativeClustering(n_clusters=max_speakers, metric="cosine",
                                            linkage="average")
            assigned = model.fit_predict(matrix[good_idx])

        for i, cluster in zip(good_idx, assigned):
            labels[keys[i]] = int(cluster)
    elif len(good_idx) == 1:
        labels[keys[good_idx[0]]] = 0

    nxt = (max(labels.values()) + 1) if labels else 0
    for i in np.where(bad)[0]:
        labels[keys[i]] = nxt
        nxt += 1

    # Number the global speakers by total appearances so SPEAKER_00 is the
    # most present voice - stable and easier to read than cluster order.
    order = sorted({v for v in labels.values()},
                   key=lambda c: -sum(1 for v in labels.values() if v == c))
    rename = {c: f"SPEAKER_{i:02d}" for i, c in enumerate(order)}
    return {k: rename[v] for k, v in labels.items()}


# ------------------------------------------------------------------- run

def run_streaming(audio_path: str, *, work_dir: str, language: str = "hi",
                  model_size: str = "tiny", beam_size: int = 5,
                  chunk_seconds: float = CHUNK_SECONDS,
                  num_speakers: int | None = None,
                  max_speakers: int | None = 6,
                  use_llm: bool = False, llm_model: str = analyze.MODEL,
                  on_event=None, on_progress=None) -> dict:
    """
    Same result as run.run(), but emits events as each chunk completes.

    on_event(dict) receives, in order:
      {"type": "meta"}                 duration, chunk count, language check
      {"type": "chunk", ...}           one per block: its utterances, provisional
      {"type": "reconciled", ...}      global speaker map; earlier labels change here
      {"type": "done"}                 final result is on disk
    """
    os.makedirs(work_dir, exist_ok=True)
    started = time.time()
    events_path = os.path.join(work_dir, "events.jsonl")
    open(events_path, "w").close()

    def emit(event: dict) -> None:
        event["t"] = round(time.time() - started, 1)
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if on_event:
            on_event(event)

    def progress(stage, fraction=0.0, note=""):
        if on_progress:
            on_progress(stage, max(0.0, min(fraction, 1.0)), note)

    # --- decode once, keep in memory; every chunk is a view on it ---------
    progress("normalize", 0.0, "decoding audio")
    samples = audio.decode(audio_path)
    wav = audio.write_wav(samples, os.path.join(work_dir, "audio_16k.wav"))
    duration = len(samples) / audio.SAMPLE_RATE
    progress("normalize", 1.0, f"{duration:.0f}s of audio")

    detected, confidence, _ = transcribe.detect_language(wav)
    mismatch = detected != language and confidence >= 0.5

    chunks = plan_chunks(samples, chunk_seconds)
    emit({"type": "meta", "duration": round(duration, 2), "n_chunks": len(chunks),
          "language": language, "detected_language": detected,
          "detected_confidence": confidence, "language_mismatch": mismatch,
          "model": model_size})

    pipeline = diarize.get_pipeline()
    all_segments, all_turns, embeddings = [], [], {}

    for i, (a, b) in enumerate(chunks):
        chunk = samples[a:b]
        offset = a / audio.SAMPLE_RATE
        label = f"chunk {i + 1}/{len(chunks)}"
        progress("chunk", i / len(chunks), f"{label} — transcribing")

        segs, _meta = transcribe.transcribe_array(
            chunk, language=language, model_size=model_size, beam_size=beam_size)

        progress("chunk", (i + 0.5) / len(chunks), f"{label} — finding speakers")
        turns, _info = diarize.diarize_array(
            chunk, pipeline=pipeline, max_speakers=max_speakers)

        # Provisional, chunk-local ids. reconcile() replaces them at the end.
        for t in turns:
            t["speaker"] = f"c{i}_{t['speaker']}"
        embeddings.update({
            f"c{i}_{k}": v for k, v in
            _speaker_embeddings(pipeline, chunk,
                                [{**t, "speaker": t["speaker"].split("_", 1)[1],
                                  "start": t["start"], "end": t["end"]}
                                 for t in turns]).items()
        })

        local = assign.assign_speakers(segs, turns)
        local_utts = assign.to_utterances(local)

        # Shift onto the lesson clock before anything leaves this loop.
        for coll in (local, turns, local_utts):
            for item in coll:
                item["start"] = round(item["start"] + offset, 2)
                item["end"] = round(item["end"] + offset, 2)

        all_segments.extend(local)
        all_turns.extend(turns)

        emit({"type": "chunk", "index": i, "of": len(chunks),
              "start": round(offset, 2), "end": round(b / audio.SAMPLE_RATE, 2),
              "utterances": local_utts, "provisional": True})
        progress("chunk", (i + 1) / len(chunks),
                 f"{label} done — {len(local_utts)} utterances")

    # --- global reconciliation -------------------------------------------
    progress("reconcile", 0.0, "matching voices across the whole lesson")
    mapping = reconcile(embeddings, max_speakers=max_speakers,
                        num_speakers=num_speakers)
    for coll in (all_segments, all_turns):
        for item in coll:
            item["speaker"] = mapping.get(item["speaker"], item["speaker"])
    all_turns.sort(key=lambda t: t["start"])
    all_segments.sort(key=lambda s: s["start"])

    speakers = sorted({t["speaker"] for t in all_turns})
    emit({"type": "reconciled", "mapping": mapping, "n_speakers": len(speakers),
          "speakers": speakers})
    progress("reconcile", 1.0, f"{len(speakers)} speakers")

    # --- everything downstream, on the reconciled whole -------------------
    progress("analyze", 0.0, "scoring the lesson")
    meta = {
        "language": language, "duration": round(duration, 2),
        "n_segments": len(all_segments), "model": model_size,
        "beam_size": beam_size, "mode": "streaming",
        "n_chunks": len(chunks), "chunk_seconds": chunk_seconds,
        "requested_language": language, "detected_language": detected,
        "detected_confidence": confidence, "language_mismatch": mismatch,
        "n_speakers": len(speakers), "speakers": speakers,
        "n_turns": len(all_turns), "diarization_model": diarize.MODEL_ID,
    }

    segments = assign.assign_speakers(all_segments, all_turns)
    utterances = assign.to_utterances(segments)
    stats = assign.speaker_stats(utterances, duration)
    verdict = roles.identify_teacher(utterances, stats, language, duration)
    utterances, names = roles.label_roles(utterances, verdict["teacher"])
    segments, _ = roles.label_roles(segments, verdict["teacher"])

    result = {
        "meta": {**meta, "elapsed": round(time.time() - started, 1),
                 "source": os.path.basename(audio_path)},
        "teacher": verdict,
        "speakers": {sp: {**s, "label": names.get(sp, sp),
                          "role": "teacher" if sp == verdict["teacher"] else "student"}
                     for sp, s in stats.items()},
        "turns": all_turns, "segments": segments, "utterances": utterances,
        **analyze.analyze(utterances, meta, language, use_llm=use_llm, model=llm_model),
    }

    with open(os.path.join(work_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    emit({"type": "done", "teacher": verdict["teacher"],
          "confidence": verdict["confidence"], "n_speakers": len(speakers)})
    progress("analyze", 1.0, "done")
    return result
