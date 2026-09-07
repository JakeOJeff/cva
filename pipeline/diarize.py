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
# enough forever.
MODEL_ID = "pyannote/speaker-diarization-community-1"


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
        pipeline = Pipeline.from_pretrained(model_id, token=_token())
        if pipeline is None:
            raise RuntimeError(
                f"could not load {model_id}. The usual cause is not having "
                "accepted the model terms on the Hub with this token's account."
            )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        _PIPELINES[model_id] = pipeline
    return _PIPELINES[model_id]


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

    output = pipeline(wav_path, **kwargs)

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
