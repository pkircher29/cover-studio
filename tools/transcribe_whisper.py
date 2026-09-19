"""Isolated Whisper large-v3 worker; timestamps retain the untrimmed audio clock."""
import argparse
import json
from pathlib import Path
from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--models", required=True)
    args = parser.parse_args()
    model = WhisperModel("large-v3", device="cpu", compute_type="int8",
                         cpu_threads=8, download_root=args.models)
    segments, info = model.transcribe(args.audio, language="en", beam_size=5,
        word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    segments = list(segments)
    result = {"model": "whisper-large-v3", "runtime": "faster-whisper",
        "device": "cpu", "compute_type": "int8", "time_origin": "original_audio",
        "duration": info.duration, "language": info.language,
        "text": " ".join(s.text.strip() for s in segments).strip(),
        "words": [{"text": w.word.strip(), "start": w.start, "end": w.end,
                   "probability": w.probability} for s in segments for w in s.words or []]}
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
