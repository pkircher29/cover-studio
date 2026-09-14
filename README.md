# Cover Studio

A local, one-click song-cover generator built on [YuE2](https://huggingface.co/m-a-p/YuE2-3B) via the
[audio.cpp](https://github.com/0xShug0/audio.cpp) inference engine.

Drop in a recording and it:
1. Transcribes the lyrics (speech-to-text, [Qwen3-ASR](https://huggingface.co/audio-cpp/audio.cpp-gguf))
2. Transcribes the melody ([SheetSage2](https://huggingface.co/m-a-p/SheetSage2))
3. Generates a finished cover for every style you multi-select from 16 presets (or write your own)
4. Saves each finished take to `completed/` the instant it's done — a crash mid-batch never loses
   already-finished styles

Extras: a "Keep Original Style" baseline preset, a major/minor key-flip toggle (edits the transcribed
ABC score's key signature), and a vocal gender hint (Any/Male/Female).

## Requirements

- Windows, with [`audio.cpp`](https://github.com/0xShug0/audio.cpp) built separately (`dev` branch,
  `windows-vulkan-release` or `windows-cpu-release` preset — see its own docs). This repo does not
  vendor that engine.
- YuE2-3B GGUF weights (`audio-cpp/Yue2-3B-GGUF` on Hugging Face) and Qwen3-ASR-0.6B GGUF weights
  (`audio-cpp/audio.cpp-gguf`), placed under the engine's `models/` directory.
- Python 3.12 for this app's own venv (`uv venv .venv`), with `transformers==4.45.2` pinned exactly —
  SheetSage2's custom modeling code breaks on transformers 5.x.
- `pip install -r requirements.txt` (torch from the CPU wheel index is enough; no GPU required for
  the app itself — YuE2 generation runs through the separately-built engine).

## Known limitation

YuE2 generation on Vulkan currently crashes on Intel Arc GPUs with a buffer-alignment assertion in
ggml-vulkan (upstream issue: [audio.cpp#535](https://github.com/0xShug0/audio.cpp/issues/535)). The
engine's `server.json` should set `"backend": "cpu"` until that's fixed upstream — slower, but correct.

## Running

Configure the engine (`server.json` next to `audiocpp_server.exe`) with `yue2` and `asr` (`qwen3_asr`)
model entries, then run `start.ps1` — it starts the engine, starts this app, and opens
http://127.0.0.1:8420.
