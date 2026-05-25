// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for gather/scatter and layer-norm kernels.

#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

#include "../inference/plugins/gather_scatter_kernels.h"
#include "../utils/error.h"

using namespace fluxrt::plugins;

static void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << err << " at " << file << ":" << line << ": "
                  << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}
#define CK(err) check_cuda(err, __FILE__, __LINE__)

template <typename T>
static std::vector<T> to_host(const T* d_ptr, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

int main() {
    std::cout << "[test_gather_scatter] Starting...\n";

    // -------------------------------------------------------------------
    // Test 1: gather_tokens
    // -------------------------------------------------------------------
    {
        int dim = 3;
        std::vector<float> h_src = {
            0,1,2,   // token 0
            3,4,5,   // token 1
            6,7,8,   // token 2
            9,10,11  // token 3
        };
        std::vector<int32_t> h_indices = {0, 2, 3};  // gather tokens 0,2,3

        float* d_src;     CK(cudaMalloc(&d_src,     h_src.size() * sizeof(float)));
        float* d_dst;     CK(cudaMalloc(&d_dst,     9 * sizeof(float)));
        int32_t* d_idx;   CK(cudaMalloc(&d_idx,     h_indices.size() * sizeof(int32_t)));

        CK(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_idx, h_indices.data(), h_indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

        gather_tokens(d_src, d_dst, d_idx, 3, dim, 0);
        CK(cudaDeviceSynchronize());

        auto h_dst = to_host(d_dst, 9);
        std::vector<float> expected = {0,1,2, 6,7,8, 9,10,11};
        bool ok = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(h_dst[i] - expected[i]) > 1e-5f) ok = false;
        }
        std::cout << "  gather_tokens: " << (ok ? "PASS" : "FAIL") << "\n";

        CK(cudaFree(d_src)); CK(cudaFree(d_dst)); CK(cudaFree(d_idx));
    }

    // -------------------------------------------------------------------
    // Test 2: scatter_tokens
    // -------------------------------------------------------------------
    {
        int seq_len = 4, dim = 3;
        std::vector<float> h_src(seq_len * dim, -1.0f);
        std::vector<float> h_gathered = {100,101,102, 200,201,202};
        std::vector<int32_t> h_indices = {1, 3};

        float* d_src;     CK(cudaMalloc(&d_src,     h_src.size() * sizeof(float)));
        float* d_gather;  CK(cudaMalloc(&d_gather,  h_gathered.size() * sizeof(float)));
        int32_t* d_idx;   CK(cudaMalloc(&d_idx,     h_indices.size() * sizeof(int32_t)));

        CK(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gather, h_gathered.data(), h_gathered.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_idx, h_indices.data(), h_indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

        scatter_tokens(d_gather, d_src, d_idx, 2, dim, 0);
        CK(cudaDeviceSynchronize());

        auto h_dst = to_host(d_src, seq_len * dim);
        bool ok = true;
        // token 0: unchanged
        for (int i = 0; i < 3; ++i) if (h_dst[i] != -1.0f) ok = false;
        // token 1: 100,101,102
        for (int i = 0; i < 3; ++i) if (h_dst[3+i] != 100.0f+i) ok = false;
        // token 2: unchanged
        for (int i = 0; i < 3; ++i) if (h_dst[6+i] != -1.0f) ok = false;
        // token 3: 200,201,202
        for (int i = 0; i < 3; ++i) if (h_dst[9+i] != 200.0f+i) ok = false;
        std::cout << "  scatter_tokens: " << (ok ? "PASS" : "FAIL") << "\n";

        CK(cudaFree(d_src)); CK(cudaFree(d_gather)); CK(cudaFree(d_idx));
    }

    // -------------------------------------------------------------------
    // Test 3: layer_norm_modulated
    // -------------------------------------------------------------------
    {
        int seq_len = 2, dim = 4;
        // x = [[1,2,3,4], [5,6,7,8]]
        std::vector<float> h_x = {1,2,3,4, 5,6,7,8};

        float* d_x;  CK(cudaMalloc(&d_x,  h_x.size() * sizeof(float)));
        float* d_y;  CK(cudaMalloc(&d_y,  h_x.size() * sizeof(float)));
        CK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

        layer_norm_modulated(d_x, d_y, nullptr, nullptr, 1e-6f, seq_len, dim, 0);
        CK(cudaDeviceSynchronize());

        auto h_y = to_host(d_y, h_x.size());
        // Row 0: mean=2.5, var=1.25, std=sqrt(1.25)≈1.118
        // normalized: [-1.3416, -0.4472, 0.4472, 1.3416]
        bool ok = true;
        float tol = 1e-4f;
        float expected0[4] = {-1.3416407f, -0.4472136f, 0.4472136f, 1.3416407f};
        for (int i = 0; i < 4; ++i) {
            if (std::fabs(h_y[i] - expected0[i]) > tol) ok = false;
        }
        // Row 1: mean=6.5, var=1.25, same normalized values
        for (int i = 0; i < 4; ++i) {
            if (std::fabs(h_y[4+i] - expected0[i]) > tol) ok = false;
        }
        std::cout << "  layer_norm_modulated: " << (ok ? "PASS" : "FAIL") << "\n";

        CK(cudaFree(d_x)); CK(cudaFree(d_y));
    }

    std::cout << "[test_gather_scatter] All tests done.\n";
    return 0;
}
