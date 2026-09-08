"""
segments + turns -> segments with a speaker on each.

Two clocks, one timeline. Whisper decides where words are, pyannote decides
where voices are, and neither agrees exactly. The rule here is majority
overlap: a segment belongs to whoever holds the most of its duration.

Segments that straddle a speaker change are the honest failure case. We
don't guess-split them (there are no word timings to split on) - we label
them with the majority speaker and record `speaker_conf` so analysis can
discount them instead of silently trusting them.
"""

CONFIDENT = 0.75      # >= this share of the segment from one speaker
NEAREST_TOLERANCE = 2.0   # seconds; how far to reach for a turn when there is no overlap


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segments, turns):
    """Returns a new list; does not mutate the input."""
    out = []
    for seg in segments:
        dur = max(seg["end"] - seg["start"], 1e-6)

        by_speaker: dict[str, float] = {}
        for t in turns:
            if t["start"] >= seg["end"]:
                break                                     # turns are sorted; nothing later can overlap
            ov = _overlap(seg["start"], seg["end"], t["start"], t["end"])
            if ov > 0:
                by_speaker[t["speaker"]] = by_speaker.get(t["speaker"], 0.0) + ov

        if by_speaker:
            speaker, covered = max(by_speaker.items(), key=lambda kv: kv[1])
            conf = round(min(covered / dur, 1.0), 3)
        else:
            speaker, conf = _nearest_speaker(seg, turns), 0.0

        out.append({
            **seg,
            "speaker": speaker,
            "speaker_conf": conf,
            "contested": len(by_speaker) > 1 and conf < CONFIDENT,
        })
    return out


def _nearest_speaker(seg, turns):
    """No overlap at all - reach for a turn just before or after, else UNKNOWN."""
    best, best_gap = None, NEAREST_TOLERANCE
    for t in turns:
        if t["end"] <= seg["start"]:
            gap = seg["start"] - t["end"]
        elif t["start"] >= seg["end"]:
            gap = t["start"] - seg["end"]
        else:
            gap = 0.0
        if gap < best_gap:
            best, best_gap = t["speaker"], gap
    return best or "UNKNOWN"


def to_utterances(segments, max_gap: float = 1.0):
    """
    Collapse consecutive same-speaker segments into utterances.

    Analysis wants turns of speech, not Whisper's ~5s chunks: "how long did
    the teacher hold the floor" is meaningless at segment granularity.
    """
    utts = []
    for seg in segments:
        if (utts and utts[-1]["speaker"] == seg["speaker"]
                and seg["start"] - utts[-1]["end"] <= max_gap):
            u = utts[-1]
            u["end"] = seg["end"]
            u["text"] = f"{u['text']} {seg['text']}".strip()
            u["n_segments"] += 1
            u["speaker_conf"] = min(u["speaker_conf"], seg.get("speaker_conf", 1.0))
        else:
            utts.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": seg["text"],
                "speaker_conf": seg.get("speaker_conf", 1.0),
                "n_segments": 1,
            })
    return utts


def speaker_stats(utterances, total_duration: float | None = None):
    """Per-speaker aggregates. The raw material for identifying the teacher."""
    stats: dict[str, dict] = {}
    for u in utterances:
        s = stats.setdefault(u["speaker"], {
            "speaker": u["speaker"], "talk_time": 0.0, "n_turns": 0,
            "n_words": 0, "first_at": u["start"], "last_at": u["end"],
            "turn_lengths": [],
        })
        dur = u["end"] - u["start"]
        s["talk_time"] += dur
        s["n_turns"] += 1
        s["n_words"] += len(u["text"].split())
        s["last_at"] = max(s["last_at"], u["end"])
        s["turn_lengths"].append(dur)

    total_talk = sum(s["talk_time"] for s in stats.values()) or 1.0
    for s in stats.values():
        s["talk_time"] = round(s["talk_time"], 2)
        s["talk_share"] = round(s["talk_time"] / total_talk, 4)
        s["avg_turn"] = round(s["talk_time"] / s["n_turns"], 2)
        s["longest_turn"] = round(max(s["turn_lengths"]), 2)
        s["span"] = round(s["last_at"] - s["first_at"], 2)
        if total_duration:
            # Clamped. Whisper's last segment can end a few seconds past the
            # decoded duration, which made a teacher "present across 103% of
            # the session" on screen. Coverage is a share of the lesson, so
            # anything over 1.0 is rounding noise, not information.
            s["coverage"] = round(min(s["span"] / total_duration, 1.0), 4)
        del s["turn_lengths"]

    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["talk_time"]))
