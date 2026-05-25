# FluxRT-CPP Build Helper (Windows)
# Automatically detects CUDA and TensorRT, sets required env vars, then runs CMake build.

param(
    [string]$Config = "Release",
    [string]$Target = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# ----------------------------------------------------------------------------
# Auto-detect CUDA Toolkit
# ----------------------------------------------------------------------------
$cudaDir = $env:CUDA_PATH
if (-not $cudaDir -or -not (Test-Path $cudaDir)) {
    $candidates = Get-ChildItem -Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\" -ErrorAction SilentlyContinue | Sort-Object Name
    if ($candidates) {
        $cudaDir = $candidates[-1].FullName
        Write-Host "[build] Auto-detected CUDA: $cudaDir"
    }
}
if (-not $cudaDir -or -not (Test-Path $cudaDir)) {
    throw "CUDA Toolkit not found. Install CUDA or set CUDA_PATH environment variable."
}

# MSBuild CUDA targets need this env var
$env:CudaToolkitDir = $cudaDir
# Also put nvcc in PATH so any subprocesses can find it
$env:PATH = "$cudaDir\bin;$env:PATH"

# ----------------------------------------------------------------------------
# Auto-detect TensorRT
# ----------------------------------------------------------------------------
$trtDir = $env:TENSORRT_ROOT
if (-not $trtDir) {
    $trtCandidates = @(
        "C:\TensorRT",
        "${env:ProgramFiles}\NVIDIA Corporation\TensorRT"
    )
    foreach ($c in $trtCandidates) {
        if (Test-Path $c) {
            $trtDir = $c
            Write-Host "[build] Auto-detected TensorRT: $trtDir"
            break
        }
    }
}
if ($trtDir) {
    $env:TENSORRT_ROOT = $trtDir
    $env:PATH = "$trtDir\bin;$env:PATH"
}

# ----------------------------------------------------------------------------
# Clean if requested
# ----------------------------------------------------------------------------
$buildDir = Join-Path $PSScriptRoot "build"
if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "[build] Cleaning build directory..."
    Remove-Item -Recurse -Force $buildDir
}

# ----------------------------------------------------------------------------
# Configure (only if build dir doesn't exist or if Clean was used)
# ----------------------------------------------------------------------------
if (-not (Test-Path "$buildDir\CMakeCache.txt") -or $Clean) {
    Write-Host "[build] Configuring CMake..."
    New-Item -ItemType Directory -Force $buildDir | Out-Null
    $configureArgs = @(
        "-B", $buildDir,
        "-S", $PSScriptRoot,
        "-DCMAKE_CUDA_COMPILER=$cudaDir/bin/nvcc.exe"
    )
    if ($trtDir) {
        $configureArgs += "-DTENSORRT_ROOT=$trtDir"
    }
    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
}

# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
Write-Host "[build] Building ($Config)..."
$buildArgs = @("--build", $buildDir, "--config", $Config, "--parallel")
if ($Target) {
    $buildArgs += @("--target", $Target)
}
& cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

Write-Host "[build] Done. Binaries are in $buildDir\bin\$Config"
