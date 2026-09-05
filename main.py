
import argparse
 
from pipeline.transcribe import normalize, transcribe, save
 
 
def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--lang", default="ml", help="ml, hi, ta, en ...")
    p.add_argument("--model", default="small", help="tiny, base, small, medium")
    p.add_argument("--out", default="segments.json")
    args = p.parse_args()
 
    print("normalizing...")
    wav = normalize(args.audio)
 
    print(f"transcribing ({args.model}, lang={args.lang})... this takes a while")
    segments, meta = transcribe(wav, language=args.lang, model_size=args.model)
 
    for s in segments:
        print(f"[{fmt(s['start'])} - {fmt(s['end'])}]  {s['text']}")
 
    save(segments, meta, args.out)
 
    talk = sum(s["end"] - s["start"] for s in segments)
    print("\n---")
    print(f"duration:   {meta['duration']:.0f}s")
    print(f"speech:     {talk:.0f}s  ({talk / meta['duration']:.0%})")
    print(f"silence:    {meta['duration'] - talk:.0f}s")
    print(f"segments:   {meta['n_segments']}")
    print(f"saved ->    {args.out}")
 
 
if __name__ == "__main__":
    main()