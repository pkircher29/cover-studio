# Cover Studio

A local, one-click song-cover generator built on [YuE2](https://huggingface.co/m-a-p/YuE2-3B) via the
[audio.cpp](https://github.com/0xShug0/audio.cpp) inference engine.

Drop in a recording and it:
1. Transcribes the lyrics (speech-to-text, [Qwen3-ASR](https://huggingface.co/audio-cpp/audio.cpp-gguf))
2. Transcribes the melody ([SheetSage2](https://huggingface.co/m-a-p/SheetSage2))
3. Generates a finished cover for every style you multi-select from 16 presets — editable, and you can add
   your own, via the "Manage styles" panel (backed by `app/styles.json`)
4. Saves each finished take to `completed/` the instant it's done — a crash mid-batch never loses
   already-finished styles

Extras: a "Keep Original Style" baseline preset, a major/minor key-flip toggle (edits the transcribed
ABC score's key signature), a vocal gender hint (Any/Male/Female), and a live VRAM gauge in the topbar
(NVIDIA GPUs only) so you can see at a glance how close a generation is to running out of graphics
memory — see [SPEC.md](SPEC.md) for how it works.

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

## GPU setup and running

Use the patched [audio.cpp fork](https://github.com/pkircher29/audio.cpp) and follow
[GPU setup and verification](docs/GPU_SETUP.md). The original unpatched Vulkan
engine can abort during YuE2 generation.

Run `start.ps1` (or the existing desktop shortcut). The launcher selects a matching
native executable, checks the selected device and both servers, and opens the app
only when ready. The top bar shows the actual configured generation backend/device.
Local preferences can be saved in the ignored `launch.local.json` file.

Verified on Intel Arc B580 (12 GB): repeated 48 kHz stereo generation, including
approximately 60-second outputs in 27 seconds at 4 synthesis steps. This is a
runtime check, not a musical-quality benchmark or a guarantee for arbitrary song
lengths. The two GPU-operation regressions also pass on GTX 1650 SUPER; full
YuE2 generation on that 4 GB card is not claimed as verified.

## Credits and licenses

Powered by **[YuE / YuE2](https://github.com/multimodal-art-projection/YuE)** and its
original authors. See the [official project](https://map-yue2.github.io/) and
[YuE2 model](https://huggingface.co/m-a-p/YuE2-3B). Native inference is provided by
[audio.cpp](https://github.com/0xShug0/audio.cpp); melody transcription uses
[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) and MERT-v2, and lyrics
transcription uses Qwen3-ASR. This is an independent community interface.

Cover Studio code: [Apache-2.0](LICENSE). **YuE2, SheetSage2 and MERT-v2 model
weights: CC BY-NC 4.0 (noncommercial).** The code license does not grant commercial
permission for those models. Models and recordings are not included in this repo.
See [NOTICE](NOTICE) and [third-party credits and license links](THIRD_PARTY_NOTICES.md).
