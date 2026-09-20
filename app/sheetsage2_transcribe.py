"""CPU-only wrapper around m-a-p/SheetSage2 (transformers, trust_remote_code).

audio.cpp's native GGUF port of SheetSage2 has no published weights yet, so
transcription runs the original PyTorch model here instead. YuE2 generation
itself stays on the native Vulkan engine (audiocpp_server) - this module only
turns a source recording into an ABC melody score for that model to condition
on.
"""
from __future__ import annotations

import threading
import copy
import importlib
import json
from pathlib import Path
from score_grid import rebuild_notation


_MODEL_ID = "m-a-p/SheetSage2"
_lock = threading.Lock()
_model = None
_inference_lock = threading.Lock()


def _load_model():
    from transformers import AutoModel
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
    import torch
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


def merge_events(vocal_events: list, mix_events: list) -> list:
    """Use absolute audio seconds; only the full mix supplies rhythm/key/chords."""
    combined = []
    for original in mix_events:
        event = copy.deepcopy(original)
        if "melody" in event["values"]:
            event["values"]["melody"] = [
                n for n in event["values"]["melody"] if int(n["track"]) == 1
            ]
        combined.append(event)
    for original in vocal_events:
        notes = [copy.deepcopy(n) for n in original["values"].get("melody", [])
                 if int(n["track"]) == 0]
        if notes:
            combined.append({"time": original["time"],
                             "global_subbeat": original["global_subbeat"],
                             "values": {"melody": notes}})
    return sorted(combined, key=lambda event: (event["time"], event["global_subbeat"]))


def transcribe_cover(vocals_path: str, source_path: str, output_dir: str, on_stage=None) -> dict:
    """One full-recording pass; use the model's melody-only export verbatim."""
    import torch
    folder = Path(output_dir) / 'direct-song'
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / 'analysis-complete.json'
    with _inference_lock, torch.inference_mode():
        if on_stage:
            on_stage('Transcribing full-song melody with SheetSage2')
        if cache.is_file():
            result = json.loads(cache.read_text(encoding='utf-8'))
        else:
            result = _load_model().transcribe(source_path, output_dir=str(folder), melody_only=True)
            if result.get('abc_error') or not result.get('abc'):
                raise RuntimeError(result.get('abc_error') or 'SheetSage2 did not export a usable score')
            stored = {k: result[k] for k in ('abc', 'events', 'duration_seconds', 'warnings') if k in result}
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(stored), encoding='utf-8')
            temporary.replace(cache)
        counts = [0, 0]
        for event in result.get('events', []):
            for note in event.get('values', {}).get('melody', []):
                track = int(note['track'])
                if track in (0, 1):
                    counts[track] += 1
        return {'abc': result['abc'], 'pipeline': 'sheetsage2-direct-full-song-v1',
                'vocal_source': 'original recording', 'instrumental_source': 'original recording',
                'vocal_notes': counts[0], 'instrumental_notes': counts[1],
                'warnings': result.get('warnings', [])}


def transcribe_cover_legacy(vocals_path: str, source_path: str, output_dir: str, on_stage=None) -> dict:
    """Combine the isolated sung melody with the original instrumental part.

    Both inputs retain the original zero point. The native exporter quantizes
    their timed notes together on the full mix's beat grid, avoiding ABC splicing
    between independently estimated tempos/keys.
    """
    import torch

    def report(stage):
        if on_stage:
            on_stage(stage)

    def analyze(model, audio, directory):
        cache = directory / "analysis-complete.json"
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        result = model.transcribe(audio, output_dir=directory, melody_only=False)
        reusable = {key: result[key] for key in ("events", "duration_seconds", "prompts", "warnings")}
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps(reusable), encoding="utf-8")
        temporary.replace(cache)
        return reusable

    folder = Path(output_dir)
    with _inference_lock, torch.inference_mode():
        report("Transcribing vocal melody")
        model = _load_model()
        # Full mode retains raw events even if vocals alone lack enough beat/key
        # information to produce their own score. We use the mix's grid below.
        vocals = analyze(model, vocals_path, folder / "vocal-melody")
        report("Transcribing instrumental melody from full song")
        full = analyze(model, source_path, folder / "full-song")
        report("Building combined vocal and instrumental score")
        events = merge_events(vocals["events"], full["events"])
        exporter = importlib.import_module(model.__class__.__module__.rsplit(".", 1)[0] + ".exports_sheetsage2")
        decoded = {"schema_version": model.tokenizer.schema_version,
                   "prompts": full["prompts"], "events": events, "has_eos": True}
        results = {}
        score_warnings = []
        for name, melody_only in (("melody", True), ("full", False)):
            exported = exporter.export_result(decoded, model.tokenizer, folder / name,
                                              full["duration_seconds"], melody_only=melody_only)
            payload = exported["payload"]
            if exported.get("abc_error") and any(reason in exported["abc_error"] for reason in (
                "cannot be represented on the decoded subbeat grid", "overlapping quantized melody notes")):
                payload["abc"], warnings = rebuild_notation(exporter, folder / name, melody_only)
                exported["abc_error"] = None
                score_warnings.extend(warnings)
            if exported.get("abc_error") or not payload.get("abc"):
                raise RuntimeError(exported.get("abc_error") or "Combined melody score is empty")
            results[name] = payload["abc"]
        metadata = {
            "vocal_source": "stems/vocals.wav", "instrumental_source": "original recording",
            "rhythm_key_chords_source": "original recording", "timeline_offset_seconds": 0,
            "vocal_notes": exported["vocal_notes"], "instrumental_notes": exported["instrumental_notes"],
            "warnings": vocals.get("warnings", []) + full.get("warnings", []) + exported.get("diagnostics", []) + score_warnings,
        }
        (folder / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"abc": results["melody"], "full_abc": results["full"], **metadata}
