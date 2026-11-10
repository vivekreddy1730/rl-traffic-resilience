<#
  setup.ps1 - one-time setup on Windows.

  Run from the project folder:
      powershell -ExecutionPolicy Bypass -File .\setup.ps1

  What it does:
    1. finds Python 3.11 (or 3.10 / 3.12)
    2. creates a virtual environment in .venv
    3. installs the pinned packages from requirements.txt (includes SUMO 1.27.1 via pip)
    4. checks SUMO, the trained models and runs a short test simulation
  Prints "SETUP OK" at the end, or the exact steps to fix the problem.
#>
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Fail($msg, $fix) {
    Write-Host ""
    Write-Host "SETUP FAILED: $msg" -ForegroundColor Red
    Write-Host "FIX: $fix" -ForegroundColor Yellow
    exit 1
}

Write-Host "== [1/4] Looking for Python 3.10 - 3.12 ==" -ForegroundColor Cyan
$pyCmd = $null
foreach ($v in @("3.11", "3.12", "3.10")) {
    try {
        & py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $pyCmd = @("py", "-$v"); break }
    } catch {}
}
if (-not $pyCmd) {
    try {
        $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($ver -in @("3.10", "3.11", "3.12")) { $pyCmd = @("python") }
    } catch {}
}
if (-not $pyCmd) {
    Fail "Python 3.10, 3.11 or 3.12 was not found (3.13+ is not supported by the pinned scikit-learn 1.5.0)." `
         "Install Python 3.11 from https://www.python.org/downloads/release/python-3119/ (tick 'Add python.exe to PATH'), open a NEW PowerShell window and run setup.ps1 again."
}
$pyExe = $pyCmd[0]
$pyArgs = @()
if ($pyCmd.Length -gt 1) { $pyArgs = $pyCmd[1..($pyCmd.Length - 1)] }
& $pyExe @pyArgs --version

Write-Host "== [2/4] Creating virtual environment .venv ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pyExe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." "Make sure Python was installed with the 'venv' component, then retry." }
} else {
    Write-Host "   .venv already exists - reusing it"
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "== [3/4] Installing packages (first time: a few minutes, ~1 GB download for PyTorch + SUMO) ==" -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --disable-pip-version-check -q
& $venvPy -m pip install -r requirements.txt --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    Fail "pip could not install the requirements." "Check your internet connection and re-run setup.ps1. See README > Troubleshooting > PyTorch install issues."
}

Write-Host "== [4/4] Checking SUMO, models and a short test simulation ==" -ForegroundColor Cyan
if ($env:SUMO_HOME) { Write-Host "   SUMO_HOME is set to: $env:SUMO_HOME (used if it contains bin\sumo.exe; otherwise the pip SUMO is used)" }
else { Write-Host "   SUMO_HOME is not set - using SUMO 1.27.1 from the eclipse-sumo pip package (this is fine)" }
& $venvPy tests\check_setup.py
if ($LASTEXITCODE -ne 0) { Fail "The setup check reported problems (see the list above)." "Follow the steps printed above, then run setup.ps1 again." }

Write-Host ""
Write-Host "SETUP OK" -ForegroundColor Green
Write-Host "Activate the environment in this window with:   .\.venv\Scripts\Activate.ps1"
Write-Host "(or call it directly:  .\.venv\Scripts\python.exe run_demo.py)"
