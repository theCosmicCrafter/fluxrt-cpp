// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Minimal device tensor wrapper for TRT I/O buffers.

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace fluxrt {

// Simple RAII wrapper for a CUDA device buffer with shape info.
struct Tensor {
    void* d_ptr = nullptr;
    std::vector<std::size_t> shape;
    std::size_t nbytes = 0;

    Tensor() = default;

    // Allocate device memory for the given shape (element size in bytes)
    Tensor(const std::vector<std::size_t>& shape_, std::size_t elem_size);

    ~Tensor();

    // Disable copy; allow move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    std::size_t numel() const;

    // Upload from host
    void upload(const void* host_ptr) const;

    // Download to host
    void download(void* host_ptr) const;
};

// RAII wrapper for a raw CUDA device pointer (e.g. temporary allocations).
// Unlike Tensor, this does not store shape info.
template <typename T>
struct CudaDevicePtr {
    T* d_ptr = nullptr;

    CudaDevicePtr() = default;
    explicit CudaDevicePtr(std::size_t n_elems) {
        cudaMalloc(reinterpret_cast<void**>(&d_ptr), n_elems * sizeof(T));
    }
    ~CudaDevicePtr() {
        if (d_ptr) cudaFree(d_ptr);
    }

    // Disable copy; allow move
    CudaDevicePtr(const CudaDevicePtr&) = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;
    CudaDevicePtr(CudaDevicePtr&& other) noexcept : d_ptr(other.d_ptr) { other.d_ptr = nullptr; }
    CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
        if (this != &other) {
            if (d_ptr) cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            other.d_ptr = nullptr;
        }
        return *this;
    }

    T* get() const { return d_ptr; }
    explicit operator bool() const { return d_ptr != nullptr; }
};

}  // namespace fluxrt
