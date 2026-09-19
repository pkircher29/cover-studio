"""Run Demucs outside the app's SheetSage2 environment."""
import asyncio
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
_lock = asyncio.Lock()


async def separate(source_path: str, output_dir: str) -> str:
    python = Path(os.environ.get("DEMUCS_PYTHON", ROOT / ".venv-demucs" /
                  ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
    if not python.is_file():
        raise RuntimeError("Demucs environment is missing. See README: Vocal separation setup.")
    folder = Path(output_dir) / "stems"
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(ROOT / "models" / "demucs"))
    async with _lock:
        with (folder / "demucs.log").open("w", encoding="utf-8") as log:
            proc = await asyncio.create_subprocess_exec(
                str(python), str(ROOT / "tools" / "separate_vocals.py"),
                source_path, str(folder), "--device", env.get("DEMUCS_DEVICE", "cpu"),
                stdout=log, stderr=log, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                code = await asyncio.wait_for(proc.wait(), timeout=7200)
            except BaseException:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                raise
        if code:
            detail = (folder / "demucs.log").read_text(encoding="utf-8", errors="replace")[-1800:]
            raise RuntimeError(f"htdemucs_ft failed: {detail}")
    vocal_path = folder / "vocals.wav"
    if not vocal_path.is_file() or vocal_path.stat().st_size < 44:
        raise RuntimeError("htdemucs_ft produced no vocal stem")
    return str(vocal_path)
