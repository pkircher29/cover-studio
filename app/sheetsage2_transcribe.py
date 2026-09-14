"""CPU-only wrapper around m-a-p/SheetSage2 (transformers, trust_remote_code).

audio.cpp's native GGUF port of SheetSage2 has no published weights yet, so
transcription runs the original PyTorch model here instead. YuE2 generation
itself stays on the native Vulkan engine (audiocpp_server) - this module only
turns a source recording into an ABC melody score for that model to condition
on.
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch
from transformers import AutoModel

_MODEL_ID = "m-a-p/SheetSage2"
_lock = threading.Lock()
_model = None


def _load_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = AutoModel.from_pretrained(
                    _MODEL_ID, trust_remote_code=True
                ).eval().to("cpu")
    return _model


def transcribe(audio_path: str, output_dir: str) -> dict:
    """Transcribe a source recording into a melody-only ABC score.

    Returns {"abc": str, "warnings": list[str]}. Raises RuntimeError if no
    usable score was produced.
    """
    model = _load_model()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        result = model.transcribe(
            audio_path, output_dir=output_dir, melody_only=True
        )

    if not result.get("abc") or result.get("abc_error"):
        raise RuntimeError(
            result.get("abc_error") or "Transcription did not produce a usable melody score"
        )

    abc_text = result["abc"]
    abc_path = Path(output_dir) / "score.abc"
    if not abc_path.exists():
        abc_path.write_text(abc_text, encoding="utf-8")

    return {"abc": abc_text, "warnings": result.get("warnings", [])}
