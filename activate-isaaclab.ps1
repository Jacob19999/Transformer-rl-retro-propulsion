param(
    # Path to the IsaacLab venv folder (defaults to ./env_isaaclab)
    [string]$EnvPath = (Join-Path $PSScriptRoot 'env_isaaclab')
)

# IMPORTANT:
# For venv-style activation to affect your current PowerShell session, you should run:
#   . .\activate-isaaclab.ps1
# (dot-source this wrapper). If you run it without the leading dot, PATH/env changes
# still apply to the current process, but interactive affordances like `deactivate`
# may not be available in the parent scope.

$activatePs1 = Join-Path $EnvPath 'Scripts\Activate.ps1'

if (!(Test-Path $activatePs1)) {
    throw "Could not find Activate.ps1 at: $activatePs1. Check -EnvPath or repo layout."
}

# Dot-source the venv activation so it can set env vars / functions for this session.
. $activatePs1

Write-Host "Activated IsaacLab environment: $EnvPath" -ForegroundColor Green
