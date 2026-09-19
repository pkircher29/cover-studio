"""Isolated htdemucs_ft worker; preserves the source audio's full timeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(0)
    print("Loading htdemucs_ft", flush=True)
    model = get_model("htdemucs_ft").eval()
    decoded = output / "separation_input.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", args.source,
        "-ac", str(model.audio_channels), "-ar", str(model.samplerate),
        "-c:a", "pcm_f32le", str(decoded),
    ], check=True, capture_output=True)
    audio, sr = sf.read(decoded, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise RuntimeError("Source audio is empty or contains invalid samples")
    wav = torch.from_numpy(audio.T.copy())
    reference = wav.mean(0)
    scale = reference.std()
    if scale < 1e-8:
        raise RuntimeError("Source audio is silent; cannot extract vocals")
    mean = reference.mean()
    print(f"Separating vocals on {args.device}", flush=True)
    with torch.inference_mode():
        stems = apply_model(model, ((wav - mean) / scale)[None],
                            device=args.device, shifts=1, split=True,
                            overlap=0.25, progress=True, num_workers=0)[0]
        stems = stems * scale + mean
    vocals = stems[model.sources.index("vocals")].cpu().numpy().T
    if vocals.shape != audio.shape or not np.isfinite(vocals).all():
        raise RuntimeError("Vocal separation returned invalid audio")
    sf.write(output / "vocals.wav", vocals, sr, subtype="FLOAT")
    (output / "separation.json").write_text(json.dumps({
        "model": "htdemucs_ft", "device": args.device,
        "sample_rate": sr, "frames": len(vocals),
        "duration_seconds": len(vocals) / sr, "timeline_offset_seconds": 0,
    }, indent=2), encoding="utf-8")
    decoded.unlink()
    print("Vocal stem ready", flush=True)


if __name__ == "__main__":
    main()
