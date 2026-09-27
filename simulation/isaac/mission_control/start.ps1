# Serves on the LAN by default; -Local restricts to this machine.
param([int]$Port = 8830, [switch]$Local)
$ErrorActionPreference = 'Stop'
$missionRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent (Split-Path -Parent $missionRoot)
$isaacPython = Join-Path $repoRoot 'env_isaaclab/Scripts/python.exe'
if (!(Test-Path -LiteralPath $isaacPython)) { throw "Isaac Python not found: $isaacPython" }
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath 'node_modules')) { npm ci; if ($LASTEXITCODE) { throw 'npm ci failed' } }
    npm run build
    if ($LASTEXITCODE) { throw 'Web build failed' }
    if (!(Test-Path -LiteralPath 'static/geometry.json')) {
        & $isaacPython export_geometry.py
        if ($LASTEXITCODE) { throw 'USD geometry export failed' }
    }
} finally { Pop-Location }
Push-Location $missionRoot
try {
    Write-Host "Mission control: http://127.0.0.1:$Port (Ctrl+C to stop service)"
    if ($Local) { & $isaacPython -m mission_control.server --port $Port } else { & $isaacPython -m mission_control.server --port $Port --lan }
} finally { Pop-Location }
