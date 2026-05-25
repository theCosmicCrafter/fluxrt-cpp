@echo off
set TENSORRT_ROOT=C:\Users\richk\CascadeProjects\TensorRT-10.16.1.11
set PATH=%TENSORRT_ROOT%\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%
set BIN=C:\Users\richk\CascadeProjects\FluxRT-CPP\build\windows-msvc-release\bin\Release
set PROJ=C:\Users\richk\CascadeProjects\FluxRT-CPP

echo ===== test_spatial_cache =====
cd /d "%PROJ%"
"%BIN%\test_spatial_cache.exe"
echo EXITCODE_SPATIAL=%ERRORLEVEL%

echo ===== test_gather_scatter =====
"%BIN%\test_gather_scatter.exe"
echo EXITCODE_GATHER=%ERRORLEVEL%

echo ===== test_identity_plugin =====
"%BIN%\test_identity_plugin.exe"
echo EXITCODE_IDENTITY=%ERRORLEVEL%

echo ===== fluxrt_spike =====
"%BIN%\fluxrt_spike.exe" "%PROJ%\engines\flux_2_klein_4b_transformer_mask.plan" "%PROJ%\spike_output.bin"
echo EXITCODE_SPIKE=%ERRORLEVEL%

echo ===== test_spatial_cache_trt =====
"%BIN%\test_spatial_cache_trt.exe" "%PROJ%\engines\flux_2_klein_4b_transformer_mask.plan"
echo EXITCODE_TRT=%ERRORLEVEL%
