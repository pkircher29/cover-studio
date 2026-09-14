$ErrorActionPreference = "Stop"

$AudioCppDir = "D:\GITHUB\audio.cpp"
$AudioCppExe = Join-Path $AudioCppDir "build\windows-vulkan-release\bin\audiocpp_server.exe"
$AppDir = "D:\GITHUB\yue-cover-studio\app"
$Python = "D:\GITHUB\yue-cover-studio\.venv\Scripts\python.exe"

if (-not (Test-Path $AudioCppExe)) {
    throw "audiocpp_server.exe not found at $AudioCppExe. Build it first."
}

$EngineLog = Join-Path $AudioCppDir "engine_live.log"
$EngineErrLog = Join-Path $AudioCppDir "engine_live_err.log"
Remove-Item $EngineLog, $EngineErrLog -ErrorAction SilentlyContinue

# Hard-pin to the Intel Arc B580 (Vulkan device 0) only. This restricts which
# GPUs ggml-vulkan even enumerates, on top of server.json's "device": 0, so
# the AMD iGPU (device 1) can never be touched or combined with it.
$env:GGML_VK_VISIBLE_DEVICES = "0"

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
