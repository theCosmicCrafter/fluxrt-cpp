// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "../inference/kernels/spatial_kv_cache.h"
#include "../utils/error.h"

using namespace fluxrt::kernels;

void assert_exact(float a, float b, const char* name) {
    if (a != b) {
        std::cerr << "Mismatch in " << name << ": " << a << " != " << b << "\n";
        exit(1);
    }
}

int main() {
    int img_len = 4;
    int txt_len = 2;
    int full_len = img_len + txt_len;
    int out_ch = 4;
    int heads = 2;
    int head_dim = 2;
    int double_layers = 1;
    int single_layers = 1;

    SpatialCache cache(img_len, txt_len, out_ch, heads, head_dim, double_layers, single_layers);

    std::vector<int32_t> mask(full_len, 0); // initial mask all 0 (skip)
    int32_t* d_mask;
    CUDA_CHECK(cudaMalloc(&d_mask, full_len * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), full_len * sizeof(int32_t), cudaMemcpyHostToDevice));

    // 1. Preprocess mask: should flip all to 2 (execute + update) since cache is invalid initially
    cache.preprocess_mask(d_mask);

    std::vector<int32_t> mask_out(full_len);
    CUDA_CHECK(cudaMemcpy(mask_out.data(), d_mask, full_len * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < full_len; ++i) {
        if (mask_out[i] != 2) {
            std::cerr << "FAIL: mask preprocess did not set to 2\n";
            return 1;
        }
    }
    std::cout << "[test_spatial_cache] Preprocess mask OK.\n";

    // 2. Output Cache sync test
    std::vector<float> pred(img_len * out_ch);
    for(size_t i = 0; i < pred.size(); ++i) pred[i] = i + 1.0f; // 1 to 16

    float* d_pred;
    CUDA_CHECK(cudaMalloc(&d_pred, img_len * out_ch * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pred, pred.data(), img_len * out_ch * sizeof(float), cudaMemcpyHostToDevice));

    cache.sync_output_cache(d_pred, d_mask);

    // Verify pred remains same (execute = true), and cache got updated
    std::vector<float> pred_out(img_len * out_ch);
    CUDA_CHECK(cudaMemcpy(pred_out.data(), d_pred, img_len * out_ch * sizeof(float), cudaMemcpyDeviceToHost));
    for(size_t i = 0; i < pred.size(); ++i) {
        assert_exact(pred_out[i], pred[i], "output_cache execution");
    }
    std::cout << "[test_spatial_cache] Initial output sync OK.\n";

    // Step 3: Now cache is valid. Pass a mask with some 0s.
    mask = {2, 2, 0, 1, 0, 2}; // txt=2,2, img=0,1,0,2
    // For image part: index 0 (skip), index 1 (exec only), index 2 (skip), index 3 (exec+update)
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), full_len * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Let's pass new pred values
    std::vector<float> new_pred(img_len * out_ch);
    for(size_t i = 0; i < new_pred.size(); ++i) new_pred[i] = (i + 1.0f) * 10.0f; // 10, 20, ... 160
    CUDA_CHECK(cudaMemcpy(d_pred, new_pred.data(), img_len * out_ch * sizeof(float), cudaMemcpyHostToDevice));

    cache.sync_output_cache(d_pred, d_mask);

    std::vector<float> new_pred_out(img_len * out_ch);
    CUDA_CHECK(cudaMemcpy(new_pred_out.data(), d_pred, img_len * out_ch * sizeof(float), cudaMemcpyDeviceToHost));

    assert_exact(new_pred_out[0], 1.0f, "idx 0 ch 0");     // mask=0 -> restored from cache
    assert_exact(new_pred_out[4], 50.0f, "idx 1 ch 0");    // mask=1 -> new_pred
    assert_exact(new_pred_out[8], 9.0f, "idx 2 ch 0");     // mask=0 -> restored from cache
    assert_exact(new_pred_out[12], 130.0f, "idx 3 ch 0");  // mask=2 -> new_pred

    std::cout << "[test_spatial_cache] Spatial cache correctly restored skipped tokens. PASS.\n";
    CUDA_CHECK(cudaFree(d_pred));
    CUDA_CHECK(cudaFree(d_mask));
    return 0;
}
