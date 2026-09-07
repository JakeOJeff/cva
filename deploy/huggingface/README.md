---
title: Classroom Voice Analysis
emoji: 🎓
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
short_description: Transcribe a lesson, separate the speakers, find the teacher
startup_duration_timeout: 1h
---

# Classroom Voice Analysis

Upload a classroom recording. It transcribes the lesson, works out who spoke
when, decides which voice is the teacher, and reports how the talking was
distributed — teacher vs student talk ratio, question count, wait time after
questions, per-student participation.

Results stream in block by block as the audio is processed, rather than
appearing all at once at the end.

**Source and full write-up:** https://github.com/JakeOJeff/cva

## Try it

Upload a **short clip — under 5 minutes**. This Space runs on 2 shared CPU
cores, where an hour of audio is an all-day job; longer uploads are refused
rather than accepted and left to rot in a queue.

Pick the language that matches your clip. Getting it wrong does not produce an
error — Whisper returns fluent nonsense — so the app checks the audio against
what you chose and says so when they disagree.

## Honest limits of this demo

- **2 shared vCPU.** A few minutes of audio takes a few minutes to process.
- **`tiny` model.** It cannot write Indian-language scripts at all — it emits
  confident English instead. The app detects that and warns you. For a real
  transcript you need `small` or larger, which is far too slow here.
- **Nothing persists.** Space storage is wiped on restart, so transcripts and
  the ~600MB of model weights go with it. The first run after a restart is
  slow while the weights download again.
- **One job at a time.** Whisper and pyannote each saturate the CPU, so the
  queue is deliberately serial.

Run it locally, or on a machine with a GPU, and none of these apply.

## How it works

```
audio → normalize → transcribe → diarize → assign → roles → analyze
```

Each stage is a pure function over the one before it, so a wrong teacher can
be corrected and every number re-derived in about a second without touching
the audio again.

The interesting problem is that speaker identity is *global* while chunked
processing is *local*: pyannote clusters within whatever audio it is handed,
so block 3's `SPEAKER_00` is unrelated to block 7's. The app diarizes each
block on its own, shows it immediately under provisional ids, takes a voice
embedding per (block, speaker), and clusters those across the whole recording
at the end to map every local id onto a real person.
