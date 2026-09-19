"""Durable word timestamps and overlap links to SheetSage2 vocal notes."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
_lock = asyncio.Lock()


def normalized(text):
    return re.findall(r"\w+", text.casefold())


def timing_view(transcription, lyrics, folder):
    if not transcription:
        return None
    current = normalized(transcription["text"]) == normalized(lyrics or "")
    notes = []
    path = Path(folder) / "scores/vocal-melody/analysis-complete.json"
    if path.is_file():
        for event in json.loads(path.read_text(encoding="utf-8"))["events"]:
            for note in event.get("values", {}).get("melody", []):
                if note.get("track") == 0:
                    notes.append({"start": event["time"], "end": note["end_time"],
                                  "pitch": note["pitch"]})
    words = []
    if current:
        for word in transcription["words"]:
            words.append({**word, "note_indices": [i for i, note in enumerate(notes)
                if min(word["end"], note["end"]) > max(word["start"], note["start"])]})
    return {"model": transcription["model"], "current": current,
            "time_origin": "original_audio", "words": words, "notes": notes,
            "message": ((f"Acoustically aligned timing · {transcription.get('review_words', 0)} words need review. "
                         if transcription.get('alignment_model') else "Whisper large-v3: estimated word timing. ") +
                        "Click to hear it; hover for vocal notes. Generation does not enforce these timestamps.")
                if current else "Lyrics changed. Saved transcription timings no longer match these words."}


async def refine(vocals_path, folder, transcription):
    if transcription.get('alignment_method') == 'ctc-forced-alignment-v1':
        return transcription
    serialized = json.dumps(transcription, sort_keys=True)
    key = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    output = Path(folder) / f'lyrics-aligned-{key}.json'
    if output.is_file():
        return json.loads(output.read_text(encoding='utf-8'))
    python = os.environ.get('ALIGNMENT_PYTHON', os.environ.get('DEMUCS_PYTHON'))
    if not python or not Path(python).is_file():
        raise RuntimeError('Alignment needs the PyTorch/torchaudio environment. Set ALIGNMENT_PYTHON or DEMUCS_PYTHON.')
    async with _lock:
        request = Path(folder) / f'lyrics-alignment-input-{key}.json'
        request.write_text(serialized, encoding='utf-8')
        with (Path(folder) / 'alignment.log').open('w', encoding='utf-8') as log:
            proc = await asyncio.create_subprocess_exec(python, str(ROOT / 'tools/align_lyrics.py'),
                vocals_path, str(request), str(output), '--models',
                os.environ.get('ALIGNMENT_MODELS', str(ROOT / 'models/alignment')),
                stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                code = await asyncio.wait_for(proc.wait(), timeout=7200)
            except BaseException:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                raise
        if code or not output.is_file():
            detail = (Path(folder) / 'alignment.log').read_text(encoding='utf-8', errors='replace')[-1500:]
            raise RuntimeError(f'Word alignment failed (exit {code}): {detail}')
    return json.loads(output.read_text(encoding='utf-8'))


async def transcribe(vocals_path, folder):
    output = Path(folder) / "lyrics-whisper-large-v3.json"
    if output.is_file():
        return await refine(vocals_path, folder, json.loads(output.read_text(encoding="utf-8")))
    python = Path(os.environ.get("WHISPER_PYTHON", ROOT / ".venv-whisper" /
                  ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
    if not python.is_file():
        raise RuntimeError("Whisper environment missing; see README Whisper setup.")
    async with _lock:
        with (Path(folder) / "whisper.log").open("w", encoding="utf-8") as log:
            proc = await asyncio.create_subprocess_exec(str(python),
                str(ROOT / "tools/transcribe_whisper.py"), vocals_path, str(output),
                "--models", os.environ.get("WHISPER_MODELS", str(ROOT / "models/whisper")),
                stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                code = await asyncio.wait_for(proc.wait(), timeout=7200)
            except BaseException:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                raise
        if code or not output.is_file():
            detail = (Path(folder) / "whisper.log").read_text(encoding="utf-8", errors="replace")[-1800:]
            raise RuntimeError(f"Whisper large-v3 failed: {detail}")
    return await refine(vocals_path, folder, json.loads(output.read_text(encoding="utf-8")))
