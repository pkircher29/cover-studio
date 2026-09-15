$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Install the app's Python environment first (see README.md)." }
& $Python -u (Join-Path $PSScriptRoot "tools\launch.py") @args
exit $LASTEXITCODE
