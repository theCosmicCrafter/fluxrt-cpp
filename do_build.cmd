@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "C:\Users\richk\CascadeProjects\FluxRT-CPP"
set TENSORRT_ROOT=C:\Users\richk\CascadeProjects\TensorRT-10.16.1.11
set VCPKG_ROOT=C:\vcpkg
echo [build] CMake configure...
cmake --preset windows-msvc-release 2>&1
if %ERRORLEVEL% neq 0 (echo CONFIGURE FAILED & exit /b 1)
echo [build] CMake build...
cmake --build build/windows-msvc-release --config Release --parallel 2>&1
if %ERRORLEVEL% neq 0 (echo BUILD FAILED & exit /b 1)
echo [build] BUILD SUCCEEDED
exit /b 0
