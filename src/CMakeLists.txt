# Copyright 2026 FluxRT-CPP Contributors
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Utilities (error macros, tensor wrappers, logging)
# ---------------------------------------------------------------------------
add_library(fluxrt_utils OBJECT
    utils/tensor.cpp
)
target_include_directories(fluxrt_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(fluxrt_utils PUBLIC
    CUDA::cudart
)
set_target_properties(fluxrt_utils PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---------------------------------------------------------------------------
# CUDA kernels (must be separate static lib to avoid nvcc mixed-source issues)
# ---------------------------------------------------------------------------
add_library(fluxrt_kernels STATIC
    inference/kernels/spatial_kv_cache.cu
)
target_include_directories(fluxrt_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(fluxrt_kernels PUBLIC
    CUDA::cudart
)
set_target_properties(fluxrt_kernels PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    COMPILE_OPTIONS ""  # Prevent MSVC /W4 /WX from being passed directly to nvcc
)

# ---------------------------------------------------------------------------
# Inference: TensorRT engine manager
# ---------------------------------------------------------------------------
add_library(fluxrt_inference OBJECT
    inference/engine_manager.cpp
)
target_include_directories(fluxrt_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)
target_link_libraries(fluxrt_inference PUBLIC
    fluxrt_utils
    fluxrt_kernels
    TensorRT::TensorRT
)
set_target_properties(fluxrt_inference PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---------------------------------------------------------------------------
# Phase 0.7 spike executable
# ---------------------------------------------------------------------------
add_executable(fluxrt_spike
    spike/main.cpp
)
target_link_libraries(fluxrt_spike PRIVATE
    fluxrt_inference
    fluxrt_utils
    TensorRT::TensorRT
)
set_target_properties(fluxrt_spike PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---------------------------------------------------------------------------
# Spatial Cache Test Harness (Phase 0.8)
# ---------------------------------------------------------------------------
add_executable(test_spatial_cache
    spike/test_spatial_cache.cu
)
target_link_libraries(test_spatial_cache PRIVATE
    fluxrt_kernels
    fluxrt_inference
    fluxrt_utils
    CUDA::cudart
)
set_target_properties(test_spatial_cache PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMPILE_OPTIONS ""  # Prevent MSVC /W4 /WX from being passed directly to nvcc
)

# ---------------------------------------------------------------------------
# Spatial Cache + TRT Integration Test (Phase 0.9)
# ---------------------------------------------------------------------------
add_executable(test_spatial_cache_trt
    spike/test_spatial_cache_trt.cpp
)
target_link_libraries(test_spatial_cache_trt PRIVATE
    fluxrt_inference
    fluxrt_utils
    fluxrt_kernels
    TensorRT::TensorRT
    CUDA::cudart
)
set_target_properties(test_spatial_cache_trt PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---------------------------------------------------------------------------
# Phase 1.0: IPluginV3 plugins
# ---------------------------------------------------------------------------
add_library(fluxrt_plugins STATIC
    inference/plugins/flux2_plugin_base.cpp
    inference/plugins/identity_plugin.cpp
    inference/plugins/plugin_registry.cpp
    inference/plugins/plugin_init.cpp
    inference/plugins/gather_scatter_kernels.cu
    inference/plugins/attention_kernels.cu
    inference/plugins/flux2_kernels.cu
    inference/plugins/weight_loader.cpp
    inference/plugins/flux2_double_block_plugin.cpp
)
target_include_directories(fluxrt_plugins PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)
target_link_libraries(fluxrt_plugins PUBLIC
    TensorRT::TensorRT
    CUDA::cudart
    CUDA::cublas
)
set_target_properties(fluxrt_plugins PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    COMPILE_OPTIONS ""  # Prevent MSVC /W4 /WX from being passed directly to nvcc
)

add_executable(test_identity_plugin
    spike/test_identity_plugin.cpp
)
target_link_libraries(test_identity_plugin PRIVATE
    fluxrt_plugins
    fluxrt_utils
    TensorRT::TensorRT
    CUDA::cudart
)
set_target_properties(test_identity_plugin PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# ---------------------------------------------------------------------------
# Phase 1.2: Gather / scatter kernel test
# ---------------------------------------------------------------------------
add_executable(test_gather_scatter
    spike/test_gather_scatter.cpp
)
target_include_directories(test_gather_scatter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_gather_scatter PRIVATE
    fluxrt_plugins
    CUDA::cudart
)
set_target_properties(test_gather_scatter PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

