// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Minimal smoke test for the IPluginV3 identity plugin.
// Builds a tiny network with one IdentityPlugin node and verifies
// that the output equals the input.

#include <iostream>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include "../inference/plugins/identity_plugin.h"
#include "../inference/plugins/plugin_init.h"
#include "../utils/error.h"

using namespace nvinfer1;

// Minimal logger for createInferBuilder / createInferRuntime
class TestLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

int main() {
    try {
        std::cout << "[test_identity_plugin] Starting...\n";
        TestLogger logger;

        // -------------------------------------------------------------------
        // Create builder and network
        // -------------------------------------------------------------------
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
        if (!builder) {
            std::cerr << "[test_identity_plugin] Failed to create builder\n";
            return 1;
        }

        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0U));
        if (!network) {
            std::cerr << "[test_identity_plugin] Failed to create network\n";
            return 1;
        }

        // -------------------------------------------------------------------
        // Create input tensor
        // -------------------------------------------------------------------
        ITensor* input = network->addInput("input", DataType::kFLOAT, Dims3{1, 4, 4});
        if (!input) {
            std::cerr << "[test_identity_plugin] Failed to add input\n";
            return 1;
        }

        // -------------------------------------------------------------------
        // Add IdentityPlugin via plugin layer
        // -------------------------------------------------------------------
        fluxrt::plugins::IdentityPluginCreator creator;
        IPluginV3* plugin = creator.createPlugin("identity", nullptr, TensorRTPhase::kBUILD);
        if (!plugin) {
            std::cerr << "[test_identity_plugin] Failed to create plugin\n";
            return 1;
        }

        std::vector<ITensor*> inputs{input};
        ILayer* layer = network->addPluginV3(inputs.data(), static_cast<int32_t>(inputs.size()),
                                             nullptr, 0, *plugin);
        if (!layer) {
            std::cerr << "[test_identity_plugin] Failed to add plugin layer\n";
            return 1;
        }

        layer->setName("identity_layer");
        ITensor* output = layer->getOutput(0);
        output->setName("output");
        network->markOutput(*output);

        std::cout << "[test_identity_plugin] Network built.\n";

        // -------------------------------------------------------------------
        // Build engine
        // -------------------------------------------------------------------
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "[test_identity_plugin] Failed to create builder config\n";
            return 1;
        }

        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 20);

        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            std::cerr << "[test_identity_plugin] Failed to build serialized network\n";
            return 1;
        }

        std::cout << "[test_identity_plugin] Engine serialized. Size=" << plan->size() << " bytes\n";

        // -------------------------------------------------------------------
        // Deserialize and run
        // -------------------------------------------------------------------
        fluxrt::plugins::initFluxRTPlugins();

        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        if (!runtime) {
            std::cerr << "[test_identity_plugin] Failed to create runtime\n";
            return 1;
        }

        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine) {
            std::cerr << "[test_identity_plugin] Failed to deserialize engine\n";
            return 1;
        }

        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        if (!context) {
            std::cerr << "[test_identity_plugin] Failed to create execution context\n";
            return 1;
        }

        // -------------------------------------------------------------------
        // Allocate and run
        // -------------------------------------------------------------------
        const int N = 16;  // 1*4*4
        std::vector<float> h_in(N);
        for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

        float* d_in;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        context->setTensorAddress("input", d_in);
        context->setTensorAddress("output", d_out);

        bool ok = context->enqueueV3(0);
        if (!ok) {
            std::cerr << "[test_identity_plugin] enqueueV3 failed\n";
            return 1;
        }

        std::vector<float> h_out(N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));

        // -------------------------------------------------------------------
        // Verify
        // -------------------------------------------------------------------
        bool pass = true;
        for (int i = 0; i < N; ++i) {
            if (h_out[i] != h_in[i]) {
                pass = false;
                std::cerr << "[test_identity_plugin] Mismatch at " << i << ": expected " << h_in[i]
                          << ", got " << h_out[i] << "\n";
                break;
            }
        }

        if (pass) {
            std::cout << "[test_identity_plugin] PASS: output matches input exactly.\n";
            return 0;
        }
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "[test_identity_plugin] ERROR: " << e.what() << "\n";
        return 1;
    }
}
