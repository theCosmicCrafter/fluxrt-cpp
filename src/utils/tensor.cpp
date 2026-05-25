// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"
#include "error.h"

#include <iostream>
#include <numeric>

namespace fluxrt {

Tensor::Tensor(const std::vector<std::size_t>& shape_, std::size_t elem_size)
    : shape(shape_) {
    std::size_t n = std::accumulate(shape.begin(), shape.end(), std::size_t(1), std::multiplies<std::size_t>());
    nbytes = n * elem_size;
    CUDA_CHECK(cudaMalloc(&d_ptr, nbytes));
}

Tensor::~Tensor() {
    if (d_ptr) {
        cudaError_t err = cudaFree(d_ptr);
        if (err != cudaSuccess) {
            std::cerr << "[Tensor] cudaFree failed: " << cudaGetErrorString(err) << "\n";
        }
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : d_ptr(other.d_ptr), shape(std::move(other.shape)), nbytes(other.nbytes) {
    other.d_ptr = nullptr;
    other.nbytes = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (d_ptr) cudaFree(d_ptr);
        d_ptr = other.d_ptr;
        shape = std::move(other.shape);
        nbytes = other.nbytes;
        other.d_ptr = nullptr;
        other.nbytes = 0;
    }
    return *this;
}

std::size_t Tensor::numel() const {
    return std::accumulate(shape.begin(), shape.end(), std::size_t(1), std::multiplies<std::size_t>());
}

void Tensor::upload(const void* host_ptr) const {
    CUDA_CHECK(cudaMemcpy(d_ptr, host_ptr, nbytes, cudaMemcpyHostToDevice));
}

void Tensor::download(void* host_ptr) const {
    CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr, nbytes, cudaMemcpyDeviceToHost));
}

}  // namespace fluxrt
