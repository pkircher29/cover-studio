$ErrorActionPreference = "Stop"

$AudioCppDir = "D:\GITHUB\audio.cpp"
$AudioCppExe = Join-Path $AudioCppDir "build\windows-cuda-release\bin\audiocpp_server.exe"
$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
if ($env:PATH -notlike "*$CudaBin*") { $env:PATH = "$CudaBin;$env:PATH" }
$AppDir = "D:\GITHUB\yue-cover-studio\app"
$Python = "D:\GITHUB\yue-cover-studio\.venv\Scripts\python.exe"

if (-not (Test-Path $AudioCppExe)) {
    throw "audiocpp_server.exe not found at $AudioCppExe. Build it first."
}

$EngineLog = Join-Path $AudioCppDir "engine_live.log"
$EngineErrLog = Join-Path $AudioCppDir "engine_live_err.log"
Remove-Item $EngineLog, $EngineErrLog -ErrorAction SilentlyContinue

# Vulkan device enumeration: 0=AMD iGPU, 1=GTX 1650 SUPER, 2=Arc B580.
# Tried pinning to the 1650 SUPER (device 1) on 2026-09-15: it hits the same
# ggml-vulkan.cpp:2020 GGML_ASSERT(get_misalign_bytes(...)==0) batched-decode
# crash as the Arc B580 (issue #535) -- so this isn't an Arc-specific bug,
# it's a ggml-vulkan bug that also kills NVIDIA's Vulkan path. Left inert
# (backend is "cpu" in server.json) until #535 lands upstream. If server.json
# ever flips back to vulkan, GGML_VK_VISIBLE_DEVICES restricts which GPU
# ggml-vulkan enumerates so it can never combine with the others.
$env:GGML_VK_VISIBLE_DEVICES = "1"

Write-Host "Starting audio.cpp engine (backend set in server.json)..."
$engine = Start-Process -FilePath $AudioCppExe -ArgumentList "--config", "server.json", "--log" -WorkingDirectory $AudioCppDir -PassThru -WindowStyle Minimized -RedirectStandardOutput $EngineLog -RedirectStandardError $EngineErrLog

Write-Host "Waiting for engine to come up..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8180/health" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Warning "Engine did not respond within 30s; continuing anyway, check its window for errors."
}

Write-Host "Starting Cover Studio app on http://127.0.0.1:8420 ..."
Set-Location $AppDir
Start-Process "http://127.0.0.1:8420"
& $Python -m uvicorn server:app --host 127.0.0.1 --port 8420

# When uvicorn exits (Ctrl+C), also stop the engine process.
if ($engine -and -not $engine.HasExited) {
    Stop-Process -Id $engine.Id -Force
}
