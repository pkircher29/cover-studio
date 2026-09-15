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

## Known limitations

- **Vulkan backend crashes on generation**, not just on Intel Arc: `ggml-vulkan.cpp:2020
  GGML_ASSERT(get_misalign_bytes(ctx, src1) == 0)` fires in the batched-decode kernel during YuE2's
  AR phase. Originally seen on an Intel Arc B580, but reproduces identically on an NVIDIA GTX 1650
  SUPER — it's a ggml-vulkan bug, not vendor-specific (upstream issue:
  [audio.cpp#535](https://github.com/0xShug0/audio.cpp/issues/535)).
- **CUDA works but needs real VRAM headroom** — more than the weights file size alone suggests. A
  clean end-to-end generation succeeded once on a 4GB card, but a second run crashed
  (`ggml.c:1671 GGML_ASSERT(ctx->mem_buffer != NULL)`) and a third failed more gracefully with an
  explicit `cudaMalloc failed: out of memory` while requesting ~2GB on top of ~1.8GB already resident.
  Root cause: the ASR model, YuE2's AR decode buffers, and its NAR/synthesis weights can all be
  resident at once, and that combined peak comfortably exceeds 4GB even though the YuE2 Q4_0 weights
  file itself is only ~2.7GB. Budget for **at least 8GB VRAM** before trusting CUDA for unattended runs;
  on tighter cards, use the gauge (see [SPEC.md](SPEC.md)) to watch headroom, and expect it may fail on
  longer generations.
- Until the Vulkan bug lands upstream, set `"backend": "cpu"` in the engine's `server.json` — slower,
  but the only backend confirmed reliable across repeated real generations.

## Running

Configure the engine (`server.json` next to `audiocpp_server.exe`) with `yue2` and `asr` (`qwen3_asr`)
model entries, then run `start.ps1` — it starts the engine, starts this app, and opens
http://127.0.0.1:8420.
