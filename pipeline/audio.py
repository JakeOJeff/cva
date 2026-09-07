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


def probe_duration(path: str) -> float | None:
    """
    Length in seconds from the container metadata, without decoding.

    Decoding a lesson to find out how long it is costs seconds and hundreds
    of megabytes; the header already knows. Returns None when the container
    does not say, which is a reason to let the file through rather than
    reject it.
    """
    try:
        with av.open(path) as container:
            if container.duration:
                return container.duration / 1_000_000        # AV_TIME_BASE
            stream = container.streams.audio[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except Exception:                                        # noqa: BLE001
        pass
    return None


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
    """
    Returns (wav_path, duration_seconds).

    Frames are resampled and written straight to the wav as they arrive,
    instead of going through `decode`. `decode` returns the whole waveform,
    which for a 64-minute lesson means a 115MB chunk list, a 115MB
    concatenated copy, and float32 copies of twice that again - peaking near
    900MB and OOM-killing a 512MB container in the pipeline's first stage.

    Nothing needs the whole waveform here. The wav on disk is what every
    later stage opens, so the peak is one frame.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    total = 0
    with av.open(path) as container, wave.open(out_path, "wb") as w:
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)

        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )

        def emit(frame):
            nonlocal total
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                w.writeframes(arr.tobytes())
                total += arr.size

        for frame in container.decode(stream):
            emit(frame)
        emit(None)          # flush whatever the resampler still holds

    if total == 0:
        os.remove(out_path)
        raise ValueError(f"no decodable audio in {path}")

    return out_path, total / SAMPLE_RATE
