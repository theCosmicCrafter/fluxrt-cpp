cmake_minimum_required(VERSION 3.28)

# ----------------------------------------------------------------------------
# vcpkg toolchain integration
# Set VCPKG_ROOT in environment, or pass -DCMAKE_TOOLCHAIN_FILE manually.
# ----------------------------------------------------------------------------
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(FluxRT_CPP
    VERSION 0.1.0
    DESCRIPTION "Real-time FLUX.2 stream editing in C++ with TensorRT"
    LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# Global settings
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Release if user didn't pick a config
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compile commands export for clangd / IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# CUDA architectures: target Ada (RTX 4090 = 89) and Blackwell (RTX 5090 = 120)
# ----------------------------------------------------------------------------
# Force-override the default that project() picks (often "native" or first GPU).
# Users can still override on the command line with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT FLUXRT_CUDA_ARCHITECTURES_OVERRIDDEN)
    set(CMAKE_CUDA_ARCHITECTURES "89;120" CACHE STRING
        "Target CUDA architectures (89=Ada RTX 4090, 120=Blackwell RTX 5090)"
        FORCE)
    set(FLUXRT_CUDA_ARCHITECTURES_OVERRIDDEN ON CACHE INTERNAL "")
endif()

message(STATUS "FluxRT-CPP: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

# ----------------------------------------------------------------------------
# Build options
# ----------------------------------------------------------------------------
option(FLUXRT_BUILD_TESTS "Build unit / parity / kernel tests" ON)
option(FLUXRT_BUILD_BENCHMARKS "Build benchmark harness" ON)
option(FLUXRT_ENABLE_SPOUT "Enable Spout texture sharing (Windows only)" OFF)
option(FLUXRT_ENABLE_LIPSYNC "Enable optional LivePortrait postprocessor" OFF)
option(FLUXRT_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

# ----------------------------------------------------------------------------
# Find CUDA
# ----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "FluxRT-CPP: CUDA Toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_DIR}")

# ----------------------------------------------------------------------------
# Find TensorRT
# Set TENSORRT_ROOT env var or pass -DTENSORRT_ROOT=...
# ----------------------------------------------------------------------------
if(NOT DEFINED TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()

if(NOT TENSORRT_ROOT)
    message(WARNING
        "TENSORRT_ROOT not set. TensorRT integration will be skipped for "
        "Phase 0 bootstrap. Set TENSORRT_ROOT to your TensorRT 10.16+ "
        "installation directory once installed.")
else()
    message(STATUS "FluxRT-CPP: TENSORRT_ROOT = ${TENSORRT_ROOT}")

    find_path(TRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT}/include
        REQUIRED)
    find_library(TRT_NVINFER_LIB
        NAMES nvinfer_10 nvinfer
        HINTS ${TENSORRT_ROOT}/lib
        REQUIRED)
    find_library(TRT_NVONNXPARSER_LIB
        NAMES nvonnxparser_10 nvonnxparser
        HINTS ${TENSORRT_ROOT}/lib
        REQUIRED)
    find_library(TRT_NVINFER_PLUGIN_LIB
        NAMES nvinfer_plugin_10 nvinfer_plugin
        HINTS ${TENSORRT_ROOT}/lib
        REQUIRED)

    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION ${TRT_NVINFER_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TRT_INCLUDE_DIR})

    add_library(TensorRT::onnxparser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::onnxparser PROPERTIES
        IMPORTED_LOCATION ${TRT_NVONNXPARSER_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TRT_INCLUDE_DIR})

    add_library(TensorRT::plugin UNKNOWN IMPORTED)
    set_target_properties(TensorRT::plugin PROPERTIES
        IMPORTED_LOCATION ${TRT_NVINFER_PLUGIN_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TRT_INCLUDE_DIR})

    add_library(TensorRT::TensorRT INTERFACE IMPORTED)
    target_link_libraries(TensorRT::TensorRT INTERFACE
        TensorRT::nvinfer
        TensorRT::onnxparser
        TensorRT::plugin
        CUDA::cudart
        CUDA::cuda_driver)
endif()

# ----------------------------------------------------------------------------
# Compiler warnings
# ----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /Zc:preprocessor)
    if(FLUXRT_WARNINGS_AS_ERRORS)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
    if(FLUXRT_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()

# ----------------------------------------------------------------------------
# Subdirectories
# Phase 0 only adds the spike harness; later phases add inference/, pipeline/, etc.
# ----------------------------------------------------------------------------
# add_subdirectory(src)

if(FLUXRT_BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Status summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "===== FluxRT-CPP ${PROJECT_VERSION} =====")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard:      ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA archs:         ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Toolkit:       ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT:           ${TENSORRT_ROOT}")
message(STATUS "  Tests:              ${FLUXRT_BUILD_TESTS}")
message(STATUS "  Benchmarks:         ${FLUXRT_BUILD_BENCHMARKS}")
message(STATUS "  Spout:              ${FLUXRT_ENABLE_SPOUT}")
message(STATUS "  Lipsync:            ${FLUXRT_ENABLE_LIPSYNC}")
message(STATUS "  Warnings as errors: ${FLUXRT_WARNINGS_AS_ERRORS}")
message(STATUS "==========================")
message(STATUS "")
