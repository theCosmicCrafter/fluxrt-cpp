// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Phase 0.9: SpatialCache + TRT integration test.
// Validates that skipped image tokens are restored from output cache.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "../inference/engine_manager.h"
#include "../inference/kernels/spatial_kv_cache.h"
#include "../utils/error.h"
#include "../utils/tensor.h"

using fluxrt::EngineManager;
using fluxrt::Tensor;
using fluxrt::kernels::SpatialCache;

// ---------------------------------------------------------------------------
// Load binary file helper
// ---------------------------------------------------------------------------
template <typename T>
bool load_raw(const std::string& path, std::vector<T>& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto pos = f.tellg();
    if (pos <= 0) return false;
    auto size = static_cast<std::streamsize>(pos);
    f.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(T));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return f.good();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <engine.plan>\n";
        return 1;
    }

    const char* plan_path = argv[1];
    const std::string fixture_dir = "tests/fixtures";

    // Model config matching Flux-2-Klein-4B
    const int txt_seq_len = 256;
    const int img_seq_len = 4096;
    const int full_seq_len = txt_seq_len + img_seq_len;
    const int out_channels = 128;
    const int num_heads = 24;
    const int head_dim = 128;
    const int num_double_layers = 5;
    const int num_single_layers = 20;

    // Number of image tokens to skip in second pass
    const int num_skip = 1024;

    try {
        std::cout << "[test_spatial_cache_trt] Loading engine: " << plan_path << "\n";
        EngineManager engine(plan_path);
        int n_io = engine.num_io_tensors();
        std::cout << "[test_spatial_cache_trt] Engine has " << n_io << " I/O tensors\n";

        // Build name-to-index map
        std::unordered_map<std::string, int> name_to_idx;
        for (int i = 0; i < n_io; ++i) {
            name_to_idx[engine.tensor_name(i)] = i;
        }

        // -------------------------------------------------------------------
        // Load fixtures
        // -------------------------------------------------------------------
        std::vector<float> hidden_states, encoder_hidden_states, timestep, guidance;
        std::vector<int64_t> img_ids, txt_ids;

        if (!load_raw(fixture_dir + "/hidden_states.bin", hidden_states) ||
            !load_raw(fixture_dir + "/encoder_hidden_states.bin", encoder_hidden_states) ||
            !load_raw(fixture_dir + "/timestep.bin", timestep) ||
            !load_raw(fixture_dir + "/img_ids.bin", img_ids) ||
            !load_raw(fixture_dir + "/txt_ids.bin", txt_ids)) {
            std::cerr << "[test_spatial_cache_trt] ERROR: Failed to load fixtures\n";
            return 1;
        }
        load_raw(fixture_dir + "/guidance.bin", guidance); // optional

        std::cout << "[test_spatial_cache_trt] Fixtures loaded OK\n";

        // -------------------------------------------------------------------
        // Pass 1: Golden run (all tokens active)
        // -------------------------------------------------------------------
        std::cout << "[test_spatial_cache_trt] Pass 1: Golden inference (all active)...\n";
        std::vector<Tensor> buffers1 = engine.allocate_buffers();

        auto upload = [&](const std::string& name, const void* host_ptr) {
            auto it = name_to_idx.find(name);
            if (it != name_to_idx.end()) {
                buffers1[it->second].upload(host_ptr);
            }
        };

        upload("hidden_states", hidden_states.data());
        upload("encoder_hidden_states", encoder_hidden_states.data());
        upload("timestep", timestep.data());
        upload("img_ids", img_ids.data());
        upload("txt_ids", txt_ids.data());
        if (!guidance.empty()) upload("guidance", guidance.data());

        engine.infer(buffers1);

        // Lookup output tensor by name instead of assuming it's last
        auto sample_it = name_to_idx.find("sample");
        if (sample_it == name_to_idx.end()) {
            std::cerr << "[test_spatial_cache_trt] ERROR: Engine has no 'sample' output tensor\n";
            return 1;
        }
        int out_idx = sample_it->second;
        auto out_shape = engine.tensor_shape("sample");
        std::size_t out_elems = std::accumulate(
            out_shape.begin(), out_shape.end(), std::size_t(1), std::multiplies<std::size_t>());

        std::vector<float> golden(out_elems);
        buffers1[out_idx].download(golden.data());
        std::cout << "[test_spatial_cache_trt] Pass 1 complete. Output elems=" << out_elems << "\n";

        // -------------------------------------------------------------------
        // Create SpatialCache and populate output cache with golden output
        // -------------------------------------------------------------------
        std::cout << "[test_spatial_cache_trt] Creating SpatialCache...\n";
        SpatialCache cache(img_seq_len, txt_seq_len, out_channels,
                           num_heads, head_dim, num_double_layers, num_single_layers);

        // Upload golden output to device, then sync to populate cache
        float* d_golden;
        CUDA_CHECK(cudaMalloc(&d_golden, out_elems * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_golden, golden.data(), out_elems * sizeof(float), cudaMemcpyHostToDevice));

        // Create all-active mask to populate cache
        std::vector<int32_t> mask_active(full_seq_len, 2);
        int32_t* d_mask;
        CUDA_CHECK(cudaMalloc(&d_mask, full_seq_len * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_mask, mask_active.data(), full_seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

        // preprocess_mask flips invalid positions to 2 (all are invalid initially)
        cache.preprocess_mask(d_mask);
        // sync_output_cache with all-active mask: prediction passes through, cache gets updated
        cache.sync_output_cache(d_golden, d_mask);
        std::cout << "[test_spatial_cache_trt] Output cache populated.\n";

        // -------------------------------------------------------------------
        // Pass 2: Modify first num_skip image tokens to garbage (0s)
        // -------------------------------------------------------------------
        std::cout << "[test_spatial_cache_trt] Pass 2: Modified inputs (skip first "
                  << num_skip << " image tokens)...\n";

        std::vector<float> hidden_states_mod = hidden_states;
        // Each image token has 128 channels. Zero out first num_skip tokens.
        for (int i = 0; i < num_skip * out_channels; ++i) {
            hidden_states_mod[i] = 0.0f;
        }

        std::vector<Tensor> buffers2 = engine.allocate_buffers();
        auto upload2 = [&](const std::string& name, const void* host_ptr) {
            auto it = name_to_idx.find(name);
            if (it != name_to_idx.end()) {
                buffers2[it->second].upload(host_ptr);
            }
        };

        upload2("hidden_states", hidden_states_mod.data());
        upload2("encoder_hidden_states", encoder_hidden_states.data());
        upload2("timestep", timestep.data());
        upload2("img_ids", img_ids.data());
        upload2("txt_ids", txt_ids.data());
        if (!guidance.empty()) upload2("guidance", guidance.data());

        engine.infer(buffers2);
        std::cout << "[test_spatial_cache_trt] Pass 2 inference complete.\n";

        // -------------------------------------------------------------------
        // Build selective mask: text=2, image[0:num_skip]=0, image[num_skip:]=2
        // -------------------------------------------------------------------
        std::vector<int32_t> mask_selective(full_seq_len, 2);
        for (int i = txt_seq_len; i < txt_seq_len + num_skip; ++i) {
            mask_selective[i] = 0; // skip first num_skip image tokens
        }
        CUDA_CHECK(cudaMemcpy(d_mask, mask_selective.data(), full_seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

        // sync_output_cache: skipped positions restored from cache, active positions keep new values
        cache.sync_output_cache(static_cast<float*>(buffers2[out_idx].d_ptr), d_mask);

        std::vector<float> output2(out_elems);
        buffers2[out_idx].download(output2.data());
        std::cout << "[test_spatial_cache_trt] Pass 2 output cache sync complete.\n";

        // -------------------------------------------------------------------
        // Verify: skipped positions match golden, active positions differ
        // -------------------------------------------------------------------
        std::cout << "[test_spatial_cache_trt] Verifying...\n";

        int skipped_match = 0;
        int skipped_total = num_skip * out_channels;
        int active_diff = 0;
        int active_total = (img_seq_len - num_skip) * out_channels;

        for (int i = 0; i < num_skip * out_channels; ++i) {
            if (output2[i] == golden[i]) skipped_match++;
        }
        for (int i = num_skip * out_channels; i < out_elems; ++i) {
            if (output2[i] != golden[i]) active_diff++;
        }

        std::cout << "[test_spatial_cache_trt] Skipped positions match: " << skipped_match
                  << "/" << skipped_total << "\n";
        std::cout << "[test_spatial_cache_trt] Active positions differ: " << active_diff
                  << "/" << active_total << "\n";

        // Check exact match for skipped positions
        if (skipped_match != skipped_total) {
            std::cerr << "[test_spatial_cache_trt] FAIL: Skipped positions do not match golden output.\n";
            CUDA_CHECK(cudaFree(d_mask));
            return 1;
        }

        // Check that active positions are different (new computation happened)
        if (active_diff == 0) {
            std::cerr << "[test_spatial_cache_trt] FAIL: Active positions identical to golden (no new computation).\n";
            CUDA_CHECK(cudaFree(d_mask));
            return 1;
        }

        std::cout << "[test_spatial_cache_trt] PASS: Output cache correctly restores skipped tokens.\n";

        // -------------------------------------------------------------------
        // Pass 3: Verify preprocess_mask semantics with reset()
        // -------------------------------------------------------------------
        std::cout << "[test_spatial_cache_trt] Pass 3: Testing preprocess_mask + reset...\n";

        // 3a. Reset cache -> all invalid. Mask all 0s -> preprocess should flip all to 2.
        cache.reset();
        std::vector<int32_t> mask_all_zero(full_seq_len, 0);
        CUDA_CHECK(cudaMemcpy(d_mask, mask_all_zero.data(), full_seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));
        cache.preprocess_mask(d_mask);

        std::vector<int32_t> mask_after_preproc(full_seq_len);
        CUDA_CHECK(cudaMemcpy(mask_after_preproc.data(), d_mask, full_seq_len * sizeof(int32_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < full_seq_len; ++i) {
            if (mask_after_preproc[i] != 2) {
                std::cerr << "[test_spatial_cache_trt] FAIL: preprocess_mask did not flip invalid position " << i << " to 2 (got " << mask_after_preproc[i] << ")\n";
                return 1;
            }
        }
        std::cout << "[test_spatial_cache_trt] Preprocess mask after reset OK (all flipped to 2).\n";

        // 3b. Populate cache for first half of image tokens only.
        // Upload a selective mask: text=2, image[0:2048]=2, image[2048:4096]=0
        std::vector<int32_t> mask_partial(full_seq_len, 0);
        for (int i = 0; i < txt_seq_len + 2048; ++i) mask_partial[i] = 2;
        CUDA_CHECK(cudaMemcpy(d_mask, mask_partial.data(), full_seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));
        cache.sync_output_cache(static_cast<float*>(buffers2[out_idx].d_ptr), d_mask);

        // Now: text tokens valid, first half of image tokens valid, second half invalid.
        // If we provide a mask of all 0s, preprocess should only flip the second half of image tokens to 2.
        std::fill(mask_all_zero.begin(), mask_all_zero.end(), 0);
        CUDA_CHECK(cudaMemcpy(d_mask, mask_all_zero.data(), full_seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));
        cache.preprocess_mask(d_mask);

        CUDA_CHECK(cudaMemcpy(mask_after_preproc.data(), d_mask, full_seq_len * sizeof(int32_t), cudaMemcpyDeviceToHost));
        // Text tokens (0..txt_seq_len-1) were never populated -> invalid -> preprocess should flip to 2
        for (int i = 0; i < txt_seq_len; ++i) {
            if (mask_after_preproc[i] != 2) {
                std::cerr << "[test_spatial_cache_trt] FAIL: preprocess_mask did not flip invalid text position " << i << " to 2 (got " << mask_after_preproc[i] << ")\n";
                return 1;
            }
        }
        // Populated image tokens (txt_seq_len .. txt_seq_len+2047) -> valid -> should stay 0
        for (int i = txt_seq_len; i < txt_seq_len + 2048; ++i) {
            if (mask_after_preproc[i] != 0) {
                std::cerr << "[test_spatial_cache_trt] FAIL: preprocess_mask flipped valid position " << i << " to " << mask_after_preproc[i] << " (should stay 0)\n";
                return 1;
            }
        }
        // Unpopulated image tokens (txt_seq_len+2048 .. end) -> invalid -> should flip to 2
        for (int i = txt_seq_len + 2048; i < full_seq_len; ++i) {
            if (mask_after_preproc[i] != 2) {
                std::cerr << "[test_spatial_cache_trt] FAIL: preprocess_mask did not flip invalid position " << i << " to 2 (got " << mask_after_preproc[i] << ")\n";
                return 1;
            }
        }
        std::cout << "[test_spatial_cache_trt] Preprocess mask with mixed valid state OK.\n";

        std::cout << "[test_spatial_cache_trt] Phase 0.9 integration test completed successfully.\n";
        CUDA_CHECK(cudaFree(d_golden));
        CUDA_CHECK(cudaFree(d_mask));
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[test_spatial_cache_trt] ERROR: " << e.what() << "\n";
        return 1;
    }
}
