# Phase 0 — Spike Results

**Date completed:** 2026-05-24
**Decision:** GO — all exit gates passed. Proceed to Phase 1.

---

## Environment

| Component | Version |
|---|---|
| OS | Windows 11 (26200) |
| GPU | NVIDIA RTX 5090 (Blackwell, sm_120) |
| CUDA Toolkit | 13.2.78 |
| TensorRT | 10.16.1.11 |
| MSVC | 14.44.35207 (VS 2022 BuildTools) |
| Model | FLUX.2-Klein-4B FP16 (`flux-2-klein-4b.safetensors`, 7.22 GB) |

---

## Exit Gate Results

| Gate | Threshold | Measured | Status |
|---|---|---|---|
| Transformer parity (PSNR) | ≥ 40 dB | **43.6 dB** | PASS |
| Spatial cache correctness | bit-exact at 0% dynamic | 131072/131072 skipped tokens restored exactly | PASS |
| LoRA refit functional | deferred to Phase 4 (pre-baked LoRA fallback pre-approved) | — | DEFERRED |
| Build reproducibility | clean `cmake --preset windows-msvc-release && cmake --build` | clean build, 0 errors | PASS |
| Identity plugin (IPluginV3) | output == input | bit-exact | PASS |
| Gather/scatter kernels | gather, scatter, layer_norm all correct | all PASS | PASS |
| Spatial cache + TRT integration | skipped tokens restored from cache | 131072/131072 correct | PASS |

---

## Risk De-risking Summary

### Risk 1: AIO / base model architecture compatible with diffusers
**Result: PASS.** Base Klein-4B single-file `.safetensors` loaded via `verify_load.py`. All component shapes match expected FLUX.2-Klein-4B architecture (5 double-stream blocks, 20 single-stream blocks).

### Risk 2: Spatial-cache patches apply to Klein-4B architecture
**Result: PASS.** `test_spatial_cache_trt` runs two engine passes, populates output cache with golden run, applies selective mask (1024 of 4096 image tokens skipped), and confirms 131072/131072 cache-restored values match golden output.

### Risk 3: AIO transformer exports to ONNX
**Result: PASS.** `flux_2_klein_4b_transformer_512x512.onnx` exported via `export_onnx.py` with ManualRMSNorm workaround for TensorRT unsupported RMSNorm op.

### Risk 4: TensorRT 10 compiles ONNX to .plan engine
**Result: PASS.** Two `.plan` engines built:
- `flux_2_klein_4b_transformer.plan` (7.75 GB, base)
- `flux_2_klein_4b_transformer_mask.plan` (7.76 GB, with mask input)
Both load and run via `trtexec --loadEngine`.

### Risk 5: C++ harness runs engine with PSNR ≥ 40 dB vs Python reference
**Result: PASS.** `fluxrt_spike.exe` runs one forward pass against `.bin` fixture files. PSNR = **43.6 dB**.

### Risk 6: Spatial KV cache ports as standalone CUDA kernel
**Result: PASS.** `test_spatial_cache` confirms:
- `preprocess_mask` flips invalid positions to `2` (execute+update)
- `sync_output_cache` restores skipped tokens from cache (bit-exact)
- Passes with both all-active and mixed-active masks

### Risk 7: LoRA engine refit spike
**Result: DEFERRED (pre-approved).** Runtime LoRA hot-swap deferred to Phase 4. Pre-baked LoRA merge is the v1 strategy (user pre-approved in plan.md).

---

## Build Artifacts

| Artifact | Location | Size |
|---|---|---|
| `fluxrt_spike.exe` | `build/windows-msvc-release/bin/Release/` | ~500 KB |
| `test_spatial_cache.exe` | `build/windows-msvc-release/bin/Release/` | ~400 KB |
| `test_gather_scatter.exe` | `build/windows-msvc-release/bin/Release/` | ~350 KB |
| `test_identity_plugin.exe` | `build/windows-msvc-release/bin/Release/` | ~450 KB |
| `test_spatial_cache_trt.exe` | `build/windows-msvc-release/bin/Release/` | ~500 KB |
| `fluxrt_plugins.lib` | `build/windows-msvc-release/src/Release/` | plugin library |
| ONNX model | `engines/onnx/flux_2_klein_4b_transformer_512x512.onnx` | ~7.1 GB |
| TRT engine (base) | `engines/flux_2_klein_4b_transformer.plan` | 7.75 GB |
| TRT engine (mask) | `engines/flux_2_klein_4b_transformer_mask.plan` | 7.76 GB |

---

## Issues Encountered & Resolutions

1. **`Flux2DoubleBlockPlugin` missing base constructor call** — Fixed by adding `Flux2PluginBase("Flux2DoubleBlockPlugin", "1", "fluxrt")` to member initializer list.
2. **Destructor defined but not declared in `.h`** — Fixed by adding `~Flux2DoubleBlockPlugin();` to the header.
3. **`configurePluginImpl` / `getFieldsToSerializeImpl` pure virtual not implemented** — Added stub implementations (pass-through, returns empty `PluginFieldCollection`).
4. **`test_gather_scatter.cpp` unused `seq_len` variable** — Removed from Test 1 scope (warnings-as-errors enabled).

---

## Next Phase

Phase 1 is GO. The full plugin-per-block architecture skeleton (`flux2_double_block_plugin`, `identity_plugin`, `gather_scatter_kernels`, `attention_kernels`, `flux2_kernels`) is built and linked. Remaining Phase 1 work:

1. Implement `Flux2SingleBlockPlugin` (mirror of double block, single-stream)
2. Wire all 5 double + 20 single blocks as TRT plugins into a complete engine
3. Benchmark full pipeline FPS vs Python FluxRT baseline
4. Implement sparse token execution path (mask-gated gather → compute → scatter)
5. Add KV cache sync across all blocks (`sync_kv_cache` per block, per timestep)

See `specs/phase-1/plan.md` (to be written at Phase 1 kickoff).
