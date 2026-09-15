# Cover Studio — Technical Spec

This is the design/implementation reference for Cover Studio: architecture, pipeline, API surface,
and the VRAM gauge. See [README.md](README.md) for setup and known limitations.

## Architecture

```
Browser (app/static/) ──HTTP/SSE──▶ FastAPI bridge (app/server.py, :8420)
                                          │
                            ┌─────────────┼──────────────────┐
                            ▼             ▼                  ▼
                     audiocpp_server  SheetSage2          nvidia-smi
                     (:8180, native)  (in-process,        (subprocess,
                     YuE2 + Qwen3-ASR  torch/CPU)          VRAM readout)
```

- **`app/server.py`** — FastAPI app. Bridges the plain browser UI to two local engines: `audiocpp_server`
  (native, built separately from [audio.cpp](https://github.com/0xShug0/audio.cpp)) for YuE2 generation
  and Qwen3-ASR lyrics transcription, and an in-process CPU call to `sheetsage2_transcribe.py` for melody
  transcription (SheetSage2 has no published GGUF port, so it runs the original PyTorch model).
- **`app/static/`** — plain HTML/CSS/vanilla JS. No build step, no framework.
- **`app/styles.json`** — the 16 style presets (`{id, label, prompt}`), human-editable directly or through
  the in-app "Manage styles" panel. `server.py` loads it at startup instead of hardcoding presets.

## Pipeline

Two-phase, so lyrics can be reviewed/corrected before anything expensive runs:

1. **`POST /api/covers/prepare`** (multipart `file`) — uploads the source recording, starts lyrics ASR
   only (Qwen3-ASR via the engine). Returns `{session_id, job_id}`.
2. Poll `GET /api/jobs/{job_id}/result` for `{"lyrics": "..."}` (or `GET /api/jobs/{job_id}/events` for
   an SSE stream). The frontend puts this in an editable textarea.
3. **`POST /api/covers/{session_id}/generate`** (form fields: `lyrics` — the possibly-edited text,
   `styles` — JSON array of preset ids, `cot`, `seed`, `num_inference_steps`, `flip_key`, `vocal`) —
   consumes the session, starts the batch. Order: SheetSage2 melody transcription (CPU, in-process) →
   sequential per-style YuE2 generation (one style failing doesn't stop the rest).
4. **`GET /api/covers/{batch_id}/events`** (SSE) — streams `{status, stage, elapsed_s, engine_active,
   styles: [{id, label, status, filename, error}]}` on each tick.
5. Each finished take is written to `completed/<source>_<style>_<batch8>.wav` **the instant it's done**
   — a crash or closed browser mid-batch never loses already-finished styles. Served at
   `/completed/<filename>`.

### Other endpoints

- `GET /api/health` — `{ok, yue2_loaded}`, proxies the engine's own `/health`.
- `GET /api/gpu/vram` — see below.
- `GET /api/styles`, `GET /api/styles/full`, `POST /api/styles`, `PUT /api/styles/{id}`,
  `DELETE /api/styles/{id}` — CRUD on `styles.json`, no restart needed (uvicorn runs without
  `--reload`, but these handlers write the file directly and the frontend re-fetches).

### Engine API notes (audio.cpp)

`audiocpp_server`'s `/v1/tasks/run` only special-cases a few top-level request fields (`lyrics`,
`seed`, `num_inference_steps`, `audio`, `text`, ...); everything else — including YuE2's `style`,
`cot`, `abc`, `cfg_scale` — must be nested under an `"options"` object. A generation response is
plain JSON with a base64-encoded WAV string under `"audio"` (not a nested `{data: ...}`).

## VRAM gauge

Added because a GPU-backend investigation kept needing to answer "how much VRAM is actually in use
right now" and there was no way to see it without alt-tabbing to a terminal. Two pieces:

**Backend — `GET /api/gpu/vram`** (`app/server.py`)

```json
{"available": true, "name": "NVIDIA GeForce GTX 1650 SUPER", "used_mib": 1755, "total_mib": 4096}
```

or `{"available": false}` if there's no NVIDIA GPU, or `nvidia-smi` can't be found/run. Implementation:

- Locates `nvidia-smi.exe` by checking the couple of well-known install paths, then falling back to
  globbing `C:\Windows\System32\DriverStore\FileRepository\*\nvidia-smi.exe` — the driver always ships
  a copy there, but the path fragment (e.g. `nvmdsi.inf_amd64_<hash>`) changes across driver updates,
  so it can't be hardcoded. The result is cached in-process after the first successful lookup.
- Runs `nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits` via
  `asyncio.create_subprocess_exec` with a 5s timeout, parses the one CSV line.
- **Important limitation:** `memory.used` only reflects memory that was *successfully* allocated. A
  failed `cudaMalloc` (the engine logs `cudaMalloc failed: out of memory` when this happens) never
  shows up here — the number simply doesn't move, because nothing was granted. So the gauge can
  legitimately sit well under 100% right up to the moment a generation fails with an OOM error; it
  shows committed usage, not attempted usage. A natural follow-up (not yet built) would be to also
  surface allocation-failure lines tailed from the engine's log, so a failed request is visible even
  when committed memory never climbs.

**Frontend — topbar needle gauge** (`app/static/index.html`, `styles.css`, `app.js`)

- SVG semicircle, three-stop linear gradient (green → yellow → red) along the arc.
- A needle (`<g id="vramNeedle">`) rotates via CSS `transform: rotate(...)`, animated with a
  `transition`. Angle formula: `angle_deg = (pct / 100) * 180 - 90`, so 0% points left (green), 50%
  points straight up (yellow), 100% points right (red) — matching the gradient direction 1:1.
- **Scales to whatever GPU is actually installed**: the gauge's 100% mark is always `total_mib` from
  the API response, never a hardcoded VRAM size, so the same code is correct on a 4GB card or a 24GB
  one.
- Polls every 1000ms (`setInterval(pollVram, 1000)`) — fast enough to visibly track a generation in
  progress. Hides itself entirely (`vramGauge.hidden = true`) when `available` is false, e.g. on a
  non-NVIDIA machine.
- Crosses a `VRAM_MAXED_PCT = 97` threshold → adds an `.is-maxed` class that pulses the percentage
  text red, as an at-a-glance "about to run out" signal.

**Cache-busting gotcha:** `index.html` loads `app.js?v=2`. Bump this query param whenever `app.js`
changes — the browser will otherwise silently keep serving a stale cached copy indefinitely (uvicorn
runs without `--reload` and `StaticFiles` sets normal HTTP caching headers), and nothing will appear
to work even though the server is serving the new file correctly.

## GPU backend findings (2026-09-15 investigation)

Full writeup lives in the commit history / project notes, but the short version, gathered by actually
running repeated real generations (not just checking that a backend initializes):

| Backend | Result |
|---|---|
| CPU | Reliable. Slow (several minutes per style). Default. |
| Vulkan | Crashes the engine process on YuE2's AR batched-decode step, on both Intel Arc and NVIDIA hardware — a real ggml-vulkan bug ([#535](https://github.com/0xShug0/audio.cpp/issues/535)), not GPU-vendor-specific. |
| CUDA | Initializes and runs correctly, but on a 4GB card the combined peak of resident ASR model + YuE2 AR buffers + YuE2 NAR weights exceeds available VRAM on anything but a short generation. Failure is inconsistent: some allocation sites check and fail gracefully (`500` with a clear error), others don't and crash the whole engine process. |

Tuning `idle_unload_ms` and adding a `min_free_memory_mb` guard in `server.json` did **not** fix the
CUDA failures — the crash happens within a single generation's own peak footprint, not from
model-reload cycling, and the memory guard silently no-ops for YuE2 (`"indeterminate footprint
(ambiguous model directory)"` in the engine log). There is no smaller GGUF quantization to fall back
to — `q4_0` is already the smallest published for `audio-cpp/Yue2-3B-GGUF`.
