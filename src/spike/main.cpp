// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Phase 0.7 spike: load TRT engine, run one forward pass, print PSNR vs Python reference.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "../inference/engine_manager.h"
#include "../utils/error.h"
#include "../utils/tensor.h"

using fluxrt::EngineManager;
using fluxrt::Tensor;

// ---------------------------------------------------------------------------
// PSNR computation (host-side, float32)
// ---------------------------------------------------------------------------
double compute_psnr(const float* a, const float* b, std::size_t n) {
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(n);
    if (mse == 0.0) return 100.0;
    return 10.0 * std::log10(1.0 / mse);  // assumes data in [0,1] range
}

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
// Generate deterministic dummy inputs matching Python fixture
// ---------------------------------------------------------------------------
struct Fixture {
    std::vector<float> hidden_states;
    std::vector<float> encoder_hidden_states;
    std::vector<float> timestep;
    std::vector<int64_t> img_ids;
    std::vector<int64_t> txt_ids;
    std::vector<float> guidance;
};

Fixture generate_fixture(const std::string& fixture_dir = "tests/fixtures") {
    Fixture f;

    std::cout << "[spike] Loading fixtures from " << fixture_dir << "...\n" << std::flush;

    bool loaded = true;
    if (!load_raw(fixture_dir + "/hidden_states.bin", f.hidden_states)) {
        std::cout << "[spike] FAILED: hidden_states.bin\n" << std::flush; loaded = false;
    }
    if (!load_raw(fixture_dir + "/encoder_hidden_states.bin", f.encoder_hidden_states)) {
        std::cout << "[spike] FAILED: encoder_hidden_states.bin\n" << std::flush; loaded = false;
    }
    if (!load_raw(fixture_dir + "/timestep.bin", f.timestep)) {
        std::cout << "[spike] FAILED: timestep.bin\n" << std::flush; loaded = false;
    }
    if (!load_raw(fixture_dir + "/img_ids.bin", f.img_ids)) {
        std::cout << "[spike] FAILED: img_ids.bin\n" << std::flush; loaded = false;
    }
    if (!load_raw(fixture_dir + "/txt_ids.bin", f.txt_ids)) {
        std::cout << "[spike] FAILED: txt_ids.bin\n" << std::flush; loaded = false;
    }

    if (loaded) {
        std::cout << "[spike] Loaded deterministic inputs from " << fixture_dir << "\n" << std::flush;
        load_raw(fixture_dir + "/guidance.bin", f.guidance); // optional
        return f;
    }
    throw std::runtime_error("Failed to load fixtures from " + fixture_dir +
                             ". Parity testing requires exact fixture match.");
}

// ---------------------------------------------------------------------------
// Save output to binary file for Python-side validation
// ---------------------------------------------------------------------------
void save_raw(const std::string& path, const float* data, std::size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(n * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <engine.plan> [output.bin]\n";
        return 1;
    }

    const char* plan_path = argv[1];
    const char* out_path = (argc >= 3) ? argv[2] : "spike_output.bin";

    try {
        std::cout << "[spike] Starting...\n" << std::flush;
        std::cout << "[spike] About to construct EngineManager...\n" << std::flush;
        std::cout << "[spike] Loading engine: " << plan_path << "\n" << std::flush;
        EngineManager engine(plan_path);
        std::cout << "[spike] Engine loaded OK\n" << std::flush;

        std::cout << "[spike] Calling num_io_tensors()...\n" << std::flush;
        int n_io = engine.num_io_tensors();
        std::cout << "[spike] Engine has " << n_io << " I/O tensors\n" << std::flush;

        // For fixed-shape engines (built with --fixed-shapes), set_input_shape
        // is not needed and can cause errors. Dynamic-shape engines only:
        // engine.set_input_shape("hidden_states", {1, 4096, 128});

        std::cout << "[spike] Printing tensor shapes...\n" << std::flush;
        for (int i = 0; i < n_io; ++i) {
            const char* name = engine.tensor_name(i);
            auto shape = engine.tensor_shape(name);
            std::cout << "  " << name << " [";
            for (std::size_t j = 0; j < shape.size(); ++j) {
                if (j) std::cout << ", ";
                std::cout << shape[j];
            }
            std::cout << "]\n";
        }

        std::cout << "[spike] Generating fixture...\n" << std::flush;
        Fixture fix = generate_fixture();

        std::cout << "[spike] Allocating device buffers...\n" << std::flush;
        std::vector<Tensor> buffers = engine.allocate_buffers();
        std::cout << "[spike] Buffers allocated OK\n" << std::flush;

        // Pre-compute name-to-index mapping to avoid O(n*m) string comparisons
        std::cout << "[spike] Building name-to-index map...\n" << std::flush;
        std::unordered_map<std::string, int> name_to_idx;
        for (int i = 0; i < n_io; ++i) {
            name_to_idx[engine.tensor_name(i)] = i;
        }
        std::cout << "[spike] Map built OK\n" << std::flush;

        // Upload inputs using the map
        std::cout << "[spike] Uploading inputs...\n" << std::flush;
        if (auto it = name_to_idx.find("hidden_states"); it != name_to_idx.end()) {
            std::cout << "[spike] Uploading hidden_states...\n" << std::flush;
            buffers[it->second].upload(fix.hidden_states.data());
            std::cout << "[spike] hidden_states uploaded\n" << std::flush;
        }
        if (auto it = name_to_idx.find("encoder_hidden_states"); it != name_to_idx.end()) {
            std::cout << "[spike] Uploading encoder_hidden_states...\n" << std::flush;
            buffers[it->second].upload(fix.encoder_hidden_states.data());
            std::cout << "[spike] encoder_hidden_states uploaded\n" << std::flush;
        }
        if (auto it = name_to_idx.find("timestep"); it != name_to_idx.end()) {
            std::cout << "[spike] Uploading timestep...\n" << std::flush;
            buffers[it->second].upload(fix.timestep.data());
            std::cout << "[spike] timestep uploaded\n" << std::flush;
        }
        if (auto it = name_to_idx.find("img_ids"); it != name_to_idx.end()) {
            std::cout << "[spike] Uploading img_ids...\n" << std::flush;
            buffers[it->second].upload(fix.img_ids.data());
            std::cout << "[spike] img_ids uploaded\n" << std::flush;
        }
        if (auto it = name_to_idx.find("txt_ids"); it != name_to_idx.end()) {
            std::cout << "[spike] Uploading txt_ids...\n" << std::flush;
            buffers[it->second].upload(fix.txt_ids.data());
            std::cout << "[spike] txt_ids uploaded\n" << std::flush;
        }
        if (auto it = name_to_idx.find("guidance"); it != name_to_idx.end()) {
            std::cout << "[spike] Uploading guidance...\n" << std::flush;
            buffers[it->second].upload(fix.guidance.data());
            std::cout << "[spike] guidance uploaded\n" << std::flush;
        }

        std::cout << "[spike] Running inference...\n";
        engine.infer(buffers);

        // -------------------------------------------------------------------
        // Find output tensor
        // -------------------------------------------------------------------
        auto sample_it = name_to_idx.find("sample");
        if (sample_it == name_to_idx.end()) {
            std::cerr << "[spike] ERROR: Engine has no 'sample' output tensor\n";
            return 1;
        }
        int out_idx = sample_it->second;

        // Download output
        auto out_shape = engine.tensor_shape("sample");
        std::size_t out_elems = std::accumulate(
            out_shape.begin(), out_shape.end(), std::size_t(1), std::multiplies<std::size_t>());
        std::vector<float> output(out_elems);
        buffers[out_idx].download(output.data());

        std::cout << "[spike] Output shape: [";
        for (std::size_t j = 0; j < out_shape.size(); ++j) {
            if (j) std::cout << ", ";
            std::cout << out_shape[j];
        }
        std::cout << "]  elems=" << out_elems << "\n";

        // Print sample values
        std::cout << "[spike] Output sample (first 10): ";
        for (std::size_t i = 0; i < std::min<std::size_t>(10, out_elems); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << "\n";

        save_raw(out_path, output.data(), out_elems);
        std::cout << "[spike] Saved raw output to: " << out_path << "\n";

        std::vector<float> python_output;
        if (!load_raw("tests/fixtures/sample.bin", python_output)) {
            std::cerr << "[spike] ERROR: Could not load Python reference output: tests/fixtures/sample.bin\n";
            return 1;
        }
        if (python_output.size() != out_elems) {
            std::cerr << "[spike] ERROR: Python output size mismatch. C++: "
                      << out_elems << ", Python: " << python_output.size() << "\n";
            return 1;
        }

        double psnr = compute_psnr(output.data(), python_output.data(), out_elems);
        std::cout << "[spike] =========================================\n";
        std::cout << "[spike] PSNR vs Python Reference: " << psnr << " dB\n";
        std::cout << "[spike] =========================================\n";
        if (psnr >= 40.0) {
            std::cout << "[spike] PASS: PSNR meets >= 40 dB threshold.\n";
        } else {
            std::cout << "[spike] FAIL: PSNR below 40 dB threshold.\n";
        }

        std::cout << "[spike] OK Phase 0.7 harness completed.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[spike] ERROR: " << e.what() << "\n";
        return 1;
    }
}
