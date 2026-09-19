$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$LocalConfig = Join-Path $PSScriptRoot "launch.local.json"
if (Test-Path $LocalConfig) {
    $LaunchSettings = Get-Content -LiteralPath $LocalConfig -Raw | ConvertFrom-Json
    if ($LaunchSettings.python) { $Python = $LaunchSettings.python }
}
if (-not (Test-Path $Python)) { throw "Install the app's Python environment first (see README.md)." }
& $Python -u (Join-Path $PSScriptRoot "tools\launch.py") @args
exit $LASTEXITCODE
