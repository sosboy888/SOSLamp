[CmdletBinding()]
param(
    [string]$OutputName = "SOSLamp"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "src/soslamp/app.py")) {
    Write-Error "Run this from the repository root."
}

# Ensure venv is active so we use local dependencies.
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "No virtual environment detected; using global Python."
}

pyinstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name $OutputName `
    --paths "src" `
    --collect-all "PySide6" `
    --collect-all "sounddevice" `
    --hidden-import "sounddevice" `
    "src/soslamp/app.py"

Write-Host "Build complete. Find the exe under dist/$OutputName/"
