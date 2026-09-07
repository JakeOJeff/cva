"""
Audio -> speaker turns.

Produces a second, independent list:
    {"start": float, "end": float, "speaker": "SPEAKER_00"}

It knows nothing about words. `assign.py` is what marries it to the
transcript. Keeping the two apart means a bad diarization never corrupts a
good transcript - you can re-run this stage alone.
"""

import os

_PIPELINES: dict[str, object] = {}

# pyannote.audio 4.x ships this as the default community checkpoint. It is
# still gated - you accept the terms once on the Hub, then a read token is
# enough forever. Override with CVA_DIARIZATION_MODEL if you have access to a
# different checkpoint (e.g. pyannote/speaker-diarization-3.1).
MODEL_ID = os.environ.get("CVA_DIARIZATION_MODEL",
                          "pyannote/speaker-diarization-community-1")


def _gated_help(model_id: str) -> str:
    return (
        f"Your HF_TOKEN works, but the account behind it is not authorized for\n"
        f"{model_id}.\n\n"
        f"  1. Open https://hf.co/{model_id} while logged in and complete the\n"
        f"     access form. It must say 'You have been granted access'.\n"
        "  2. Check https://hf.co/settings/tokens - the token has to belong to\n"
        "     the SAME account that was granted access.\n"
        "  3. If it is a fine-grained token, enable 'Read access to contents of\n"
        "     all public gated repos you can access'. This is the usual cause:\n"
        "     fine-grained tokens cannot read gated repos by default, even when\n"
        "     the account has access. A plain Read token works without it.\n\n"
        "  Already have access to a different checkpoint? Set\n"
        "  CVA_DIARIZATION_MODEL=pyannote/speaker-diarization-3.1 in your .env."
    )


def _token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        raise RuntimeError(
            "No HF_TOKEN in the environment.\n"
            f"  1. Accept the terms at https://hf.co/{MODEL_ID}\n"
            "  2. Create a read token at https://hf.co/settings/tokens\n"
            "  3. Put HF_TOKEN=hf_... in your .env"
        )
    return tok


def get_pipeline(model_id: str = MODEL_ID):
    if model_id not in _PIPELINES:
        import torch
        from pyannote.audio import Pipeline

        # pyannote 4.x renamed this argument from `use_auth_token` to `token`.
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=_token())
        except Exception as exc:                              # noqa: BLE001
            # huggingface_hub raises GatedRepoError for "you are not in the
            # authorized list" and a 401 for a bad token. Both arrive here as
            # a wall of HTTP detail that says nothing about what to actually
            # do, so translate them.
            name = type(exc).__name__
            if "Gated" in name or "403" in str(exc):
                raise RuntimeError(_gated_help(model_id)) from exc
            if "401" in str(exc) or "Unauthorized" in name:
                raise RuntimeError(
                    "HF_TOKEN was rejected by HuggingFace. Check it is copied "
                    "whole and has not been revoked: https://hf.co/settings/tokens"
                ) from exc
            raise

        if pipeline is None:
            # from_pretrained returns None rather than raising when the config
            # loads but the pipeline cannot be built.
            raise RuntimeError(
                f"could not load {model_id}.\n\n" + _gated_help(model_id)
            )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        _PIPELINES[model_id] = pipeline
    return _PIPELINES[model_id]


def _as_waveform(wav_path: str) -> dict:
    """
    Hand pyannote the samples, not the path.

    pyannote 4.x decodes audio through torchcodec, which needs the system
    FFmpeg shared libraries - the exact dependency this project avoids by
    decoding through PyAV. Given a file path it dies with a wall of DLL
    loading errors on any machine without a full-shared FFmpeg build.

    Its own error message names the way out: pass
    {"waveform": (channel, time) tensor, "sample_rate": int} and the decoder
    is never reached. We already have 16k mono samples from audio.py, so this
    is both the fix and the shorter path.
    """
    import torch

    from .audio import SAMPLE_RATE, decode

    samples = decode(wav_path)
    return {
        "waveform": torch.from_numpy(samples).unsqueeze(0),   # (1, time)
        "sample_rate": SAMPLE_RATE,
        "uri": os.path.splitext(os.path.basename(wav_path))[0],
    }


def diarize(wav_path: str, num_speakers: int | None = None,
            min_speakers: int | None = None, max_speakers: int | None = None,
            hook=None, model_id: str = MODEL_ID):
    """
    Returns (turns, info).

    Pass num_speakers only when you actually know it. In a classroom you
    usually don't - a max_speakers ceiling is the safer hint, because
    unbounded clustering will happily split one loud teacher into three.
    """
    pipeline = get_pipeline(model_id)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
    if hook is not None:
        kwargs["hook"] = hook

    output = pipeline(_as_waveform(wav_path), **kwargs)

    # 4.x returns a DiarizeOutput; 3.x (and `legacy=True`) returns the
    # Annotation directly. We want the inclusive diarization either way -
    # overlapped speech is real speech, and `assign` resolves it by majority.
    annotation = getattr(output, "speaker_diarization", output)

    turns = [
        {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "speaker": speaker,
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])

    speakers = sorted({t["speaker"] for t in turns})
    info = {
        "n_speakers": len(speakers),
        "speakers": speakers,
        "n_turns": len(turns),
        "diarization_model": model_id,
    }
    return turns, info


if __name__ == "__main__":
    # python -m pipeline.diarize
    #
    # Verifies the token and the model access before you spend an hour
    # transcribing only to fall over at the diarize step.
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    print(f"checking access to {MODEL_ID} ...\n")
    try:
        get_pipeline()
    except RuntimeError as exc:
        print(f"FAILED\n\n{exc}")
        sys.exit(1)
    print("OK - token valid and model downloaded. Diarization will run.")
