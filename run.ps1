param(
    [string]$InputPath = ".\input",
    [string]$Config = ".\config.toml",
    [switch]$ExtractOnly,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $Root ".venv"
$VenvPythonExe = Join-Path $VenvPath "Scripts\python.exe"
$SystemPythonExe = (Get-Command python -ErrorAction Stop).Source
$RequirementsPath = Join-Path $Root "requirements.txt"

function Invoke-Step {
    param(
        [string]$Label,
        [string]$FilePath,
        [string[]]$Arguments
    )

    Write-Host "==> $Label"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Test-PythonCommand {
    param(
        [string]$PythonExe,
        [string[]]$Arguments
    )

    if (-not (Test-Path $PythonExe)) {
        return $false
    }

    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $PythonExe
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $StartInfo.Arguments = (($Arguments | ForEach-Object {
                if ($_ -match '[\s"]') {
                    '"' + $_.Replace('"', '\"') + '"'
                }
                else {
                    $_
                }
            }) -join ' ')

    $Process = [System.Diagnostics.Process]::Start($StartInfo)
    $null = $Process.StandardOutput.ReadToEnd()
    $null = $Process.StandardError.ReadToEnd()
    $Process.WaitForExit()
    return $Process.ExitCode -eq 0
}

function Test-PythonDependencies {
    param([string]$PythonExe)

    return Test-PythonCommand $PythonExe @("-c", "import cv2, numpy, requests, tqdm")
}

if (-not (Test-Path $VenvPath)) {
    try {
        Invoke-Step "Creating virtual environment" $SystemPythonExe @("-m", "venv", $VenvPath)
    }
    catch {
        Write-Warning "Virtual environment bootstrap failed. Falling back to the current Python environment."
    }
}

$PythonExe = $VenvPythonExe
$UsingVenv = Test-PythonCommand $VenvPythonExe @("-m", "pip", "--version")

if (-not $UsingVenv) {
    $PythonExe = $SystemPythonExe
    Write-Warning "The .venv interpreter is unavailable or missing pip. Using $PythonExe instead."
}

if (-not $SkipInstall) {
    $HasDependencies = Test-PythonDependencies $PythonExe

    if (-not $HasDependencies) {
        if (-not $UsingVenv) {
            Write-Warning "Installing requirements into the current Python environment because .venv is not usable."
        }
        Invoke-Step "Upgrading pip" $PythonExe @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Step "Installing project requirements" $PythonExe @("-m", "pip", "install", "-r", $RequirementsPath)
    }
    else {
        Write-Host "==> Python dependencies already available"
    }
}

if (-not (Test-PythonDependencies $PythonExe)) {
    throw "Required Python packages are missing for $PythonExe. Re-run without -SkipInstall or install from $RequirementsPath."
}

$Args = @(
    (Join-Path $Root "analyze_video.py"),
    "--input", $InputPath,
    "--config", $Config
)

if ($ExtractOnly) {
    $Args += "--extract-only"
}

Invoke-Step "Starting analyzer" $PythonExe $Args
