"""
Any audio file -> 16kHz mono float32.

Decoding goes through PyAV (bundled with faster-whisper), so there is no
dependency on a system ffmpeg binary. Every later stage assumes 16k mono:
faster-whisper wants it, pyannote wants it, and the offsets in `segments`
are only comparable if both stages saw the same waveform.
"""

import os
import wave

import av
import numpy as np

SAMPLE_RATE = 16000


def decode(path: str) -> np.ndarray:
    """Decode any container/codec to mono float32 in [-1, 1] at 16 kHz."""
    with av.open(path) as container:
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )

        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        # Flush whatever the resampler is still holding.
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError(f"no decodable audio in {path}")

    pcm = np.concatenate(chunks)
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def write_wav(samples: np.ndarray, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pcm16 = (samples.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    return out_path


def normalize(path: str, out_path: str) -> tuple[str, float]:
    """Returns (wav_path, duration_seconds)."""
    samples = decode(path)
    write_wav(samples, out_path)
    return out_path, len(samples) / SAMPLE_RATE
