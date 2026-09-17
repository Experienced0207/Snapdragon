<#
.SYNOPSIS
    SignBridge Windows 11 ARM64 / Qualcomm Snapdragon X Elite Environment Bootstrap.
.DESCRIPTION
    Bootstraps the native Windows ARM64 environment for SignBridge on HP Snapdragon laptops.
    Verifies and installs native ARM64 Node.js, Python, creates the Python virtual environment,
    installs onnxruntime-qnn for the Qualcomm Hexagon NPU, and builds frontend dependencies.
.NOTES
    Target Hardware: Qualcomm Snapdragon X Elite / Hexagon NPU (Windows 11 ARM64)
#>

[CmdletBinding()]
param (
    [switch]$SkipWingetInstall = $false
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n[SignBridge] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-WarningMsg {
    param([string]$Message)
    Write-Host "  [WARN] $Message" -ForegroundColor Yellow
}

Write-Host "======================================================================" -ForegroundColor DarkCyan
Write-Host "  SignBridge Windows 11 ARM64 (Qualcomm Snapdragon) Setup Script       " -ForegroundColor Cyan
Write-Host "  Target: HP Snapdragon Laptop / Hexagon NPU Local Acceleration Engine" -ForegroundColor DarkCyan
Write-Host "======================================================================" -ForegroundColor DarkCyan

# Detect Architecture
$arch = $env:PROCESSOR_ARCHITECTURE
Write-Step "Detecting System Hardware Architecture: $arch"
if ($arch -ne "ARM64") {
    Write-WarningMsg "Detected non-ARM64 architecture ($arch). Continuing in compatibility mode."
} else {
    Write-Success "Native ARM64 architecture confirmed."
}

# 1. Verify / Install Native ARM64 Python
Write-Step "Checking for Python ARM64 installation..."
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

if (-not $pythonCmd) {
    if ($SkipWingetInstall) {
        throw "Python not found. Please install native Python ARM64 from python.org or run without -SkipWingetInstall."
    }
    Write-Host "Python not detected. Installing Python ARM64 via winget..." -ForegroundColor Yellow
    winget install Python.Python.3.11 --architecture arm64 --scope machine --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    $pyVersion = python --version
    Write-Success "Python is installed: $pyVersion"
}

# 2. Verify / Install Native ARM64 Node.js
Write-Step "Checking for Node.js ARM64 installation..."
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue

if (-not $nodeCmd) {
    if ($SkipWingetInstall) {
        throw "Node.js not found. Please install native Node.js ARM64 from nodejs.org or run without -SkipWingetInstall."
    }
    Write-Host "Node.js not detected. Installing Node.js LTS ARM64 via winget..." -ForegroundColor Yellow
    winget install OpenJS.NodeJS.LTS --architecture arm64 --scope machine --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    $nodeVersion = node -v
    Write-Success "Node.js is installed: $nodeVersion"
}

# 3. Setup Backend Virtual Environment & onnxruntime-qnn
Write-Step "Configuring Python virtual environment for Snapdragon Hexagon NPU..."
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path "$scriptDir\..\.."
$backendDir = "$rootDir\apps\backend"
$venvDir = "$backendDir\.venv"

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment at $venvDir..."
    python -m venv $venvDir
}

$venvPython = "$venvDir\Scripts\python.exe"
$venvPip = "$venvDir\Scripts\pip.exe"

Write-Step "Upgrading pip & installing onnxruntime-qnn for Hexagon NPU..."
& $venvPython -m pip install --upgrade pip setuptools wheel

$reqFile = "$scriptDir\requirements_snapdragon.txt"
if (Test-Path $reqFile) {
    Write-Host "Installing dependencies from $reqFile..."
    & $venvPip install -r $reqFile
} else {
    Write-Host "Installing direct packages (onnxruntime-qnn, fastapi, uvicorn, numpy)..."
    & $venvPip install onnxruntime-qnn fastapi uvicorn scipy numpy mediapipe opencv-python pydantic qai-hub
}
Write-Success "Python environment and Qualcomm QNN execution provider installed successfully."

# 4. Install Desktop Frontend Dependencies
Write-Step "Installing Tauri/React frontend dependencies in apps/desktop..."
$desktopDir = "$rootDir\apps\desktop"

if (Test-Path $desktopDir) {
    Push-Location $desktopDir
    try {
        Write-Host "Running 'npm install' in $desktopDir..."
        npm install
        Write-Success "Frontend npm packages installed successfully."
    } finally {
        Pop-Location
    }
} else {
    Write-WarningMsg "apps/desktop directory not found at $desktopDir"
}

# 5. Output Final Confirmation Message
Write-Host "`n======================================================================" -ForegroundColor DarkCyan
Write-Host "SignBridge Snapdragon Environment Ready. Run 'npm run tauri build' to compile the native ARM64 executable." -ForegroundColor Green
Write-Host "======================================================================`n" -ForegroundColor DarkCyan
