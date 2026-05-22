# Phase 0 — Tasks

Spec-kit style task list. `[P]` = parallelizable with previous task.

---

## 0.1 Repo Bootstrap

- [x] Create `FluxRT-CPP/` skeleton (dirs + base files)
- [x] Write `CONSTITUTION.md`
- [x] Write `BLOCKED.md`
- [x] Write `.gitignore`, `LICENSE`, `README.md`
- [x] Write `vcpkg.json` manifest with locked baseline
- [x] Write top-level `CMakeLists.txt` (Phase 0 scope only)
- [ ] User creates empty private GitHub repo `FluxRT-CPP`
- [ ] `git init` + initial commit + push

## 0.2 Install Verification

- [x] CUDA Toolkit installed (CUDA 13.2 confirmed via `nvcc --version`)
- [ ] Visual Studio Build Tools installed (in progress)
- [ ] Verify `cl.exe` available in Developer Command Prompt
- [ ] User: download TensorRT 10.16.1 ZIP for Windows + CUDA 13.x
- [ ] User: extract to `C:\TensorRT-10.16.1`, add to PATH, set `TENSORRT_ROOT`
- [ ] Verify `trtexec --help` works
- [ ] Run `cmake -S . -B build` and confirm CUDA + TensorRT detected

## 0.3 Reference Repo Study

- [x] Clone all 6 reference repos to `c:/Users/richk/refs/`
- [ ] Read `stable-diffusion.cpp/examples/cli/main.cpp` and FLUX-related code
- [ ] Read `NVIDIA-TensorRT/demo/Diffusion/utilities.py` for refit pattern
- [ ] Read `Torch-TensorRT/examples/apps/flux_demo.py` for export pattern
- [ ] Document findings in `specs/phase-0-spike/research.md`

## 0.4 Download + Verify AIO + Capture Reference

### 0.4a — Download AIO from CivitAI
- [ ] User: create CivitAI account at https://civitai.com (will redirect to civitai.red for AIO)
- [ ] User: generate API key at https://civitai.com/user/account
- [ ] User: provide API key to Cascade (or set `CIVITAI_API_KEY` env var)
- [x] Write `tools/download_aio.py` (CivitAI API downloader)
- [ ] Run `python tools/download_aio.py` → `models/aio/flux2-klein-aio.safetensors`

### 0.4b — Verify AIO Architecture (GO/NO-GO #1)
- [ ] Write `tools/verify_aio_load.py` — load AIO via `Flux2KleinPipeline.from_single_file()` and print component shapes
- [ ] Run it; **GATE:** All three components (transformer / vae / text_encoder) load successfully?
  - YES → proceed
  - NO → diffusers may not yet support AIO single-file format; fall back to manual safetensors parse + component-by-component construction. Update BLOCKED.md.
- [ ] Verify FluxRT's `transformer_flux2.py` patches still apply to AIO (try running upstream FluxRT with AIO weights swapped in). **GATE:** end-to-end Python pipeline works?
  - YES → proceed
  - NO → AIO has different layer structure than base Klein. Major rethink: switch to base Klein + accept slower step count, OR write new patches for AIO. Pause and discuss with user.

### 0.4c — Capture AIO Reference Latents
- [ ] Write `tools/capture_reference.py` — captures (input, prompt, seed, output_latent, output_image) tuples using AIO scheduler
- [ ] Use 4-step and 6-step configs (since AIO is distilled to that range)
- [ ] Run script, save 10 fixtures to `tests/fixtures/`

## 0.5 ONNX Export — GO/NO-GO #2

- [x] Write `tools/export_onnx.py` (refactored for AIO single-file)
- [ ] [P] Adapt patterns from `Torch-TensorRT/examples/apps/flux_demo.py`
- [ ] Use FP16, batch=1, fixed shape 512×512 for spike (or BF16 if AIO precision matters)
- [ ] Run export, verify `flux2_aio_transformer_512x512_fp16.onnx` opens in Netron
- [ ] **GATE:** Does ONNX export succeed without errors?
  - YES → proceed to 0.6
  - NO → debug; if FLUX.2 has TRT-unsupported custom ops, write findings to `BLOCKED.md` and consider Torch-TensorRT mixed-mode fallback

## 0.6 TensorRT Engine Build — GO/NO-GO #2

- [ ] Write `tools/build_engine.py` — builds TRT engine from ONNX
- [ ] Use `trtexec --onnx=... --saveEngine=... --fp16` for first attempt
- [ ] Save `.plan` to `engines/flux2_transformer_fp16.plan`
- [ ] **GATE:** Does engine build succeed and `trtexec --loadEngine=...` work?
  - YES → proceed to 0.7
  - NO → debug; check op support matrix; consider engine plugin authoring or TRT version bump

## 0.7 C++ Engine Harness — GO/NO-GO #3

- [ ] Write `src/inference/engine_manager.{h,cpp}` — minimal load/run wrapper
- [ ] Write `src/utils/error.{h,cpp}` — `CUDA_CHECK`, `TRT_CHECK` macros
- [ ] Write `src/utils/tensor.{h,cpp}` — thin device-memory view
- [ ] Write `tools/spike_main.cpp` — load engine, run forward pass on fixture, print PSNR
- [ ] [P] Add `tools/CMakeLists.txt` to build the spike executable
- [ ] Run on 10 fixtures, compute PSNR vs reference latents
- [ ] **GATE:** Mean PSNR ≥ 40 dB?
  - YES → proceed to 0.8
  - NO → investigate FP16 precision loss, consider FP32 baseline first

## 0.8 Spatial KV Cache CUDA Port — GO/NO-GO #4

- [ ] Read `src/fluxrt/stream_processor/transformer_flux2.py` `SpatialCache` class deeply
- [ ] Write `src/inference/kernels/spatial_kv_cache.cu` — standalone CUDA kernel
- [ ] Write `tests/kernels/test_spatial_kv_cache.cpp` — Catch2 test, bit-exact at 0% dynamic
- [ ] [P] Run under `compute-sanitizer --tool memcheck` — must be clean
- [ ] [P] Run under `compute-sanitizer --tool racecheck` — must be clean
- [ ] **GATE:** Bit-exact output at 0% dynamic + sanitizer clean?
  - YES → proceed to 0.9
  - NO → debug kernel; or fall back to keeping cache logic in PyTorch via `torch.compile` for v1

## 0.9 LoRA Engine Refit Spike — GO/NO-GO #5

- [ ] User: provide 1 FLUX.2-Klein-4B compatible LoRA file (or download from HuggingFace)
- [ ] Rebuild engine with `kREFIT` flag (`immutable_weights=False`)
- [ ] Write `src/inference/lora_refit.{h,cpp}` — calls `IRefitter::setNamedWeights`
- [ ] Run inference with base engine, capture output A
- [ ] Refit with LoRA, run inference, capture output B
- [ ] **GATE:** PSNR(A, B) ≤ 30 dB? (low PSNR = output diverged = refit worked)
  - YES → proceed to 0.10
  - NO → drop LoRA hot-swap from v1, switch to pre-baked LoRA strategy

## 0.10 Phase 0 Results & Go/No-Go

- [ ] Write `specs/phase-0-spike/results.md` — measurements, gate outcomes, decision
- [ ] Run benchmark suite if all gates passed (FPS comparison vs Python baseline)
- [ ] Update `BLOCKED.md` — clear resolved blockers
- [ ] Update `CONSTITUTION.md` if any principles need adjustment
- [ ] User review: approve proceeding to Phase 1, or escalate

---

## Parallel Opportunities

These can be done in parallel by Cascade alongside the main sequence:

- [P] Set up `tools/run_benchmark.sh` skeleton (used Phase 1+)
- [P] Write `src/utils/log.{h,cpp}` (spdlog wrapper)
- [P] Write `src/runtime/cuda_arena.{h,cpp}` (memory pool wrapper)
- [P] Document architecture in `docs/architecture.md`

These can be done in parallel by user:

- [P] Calibration dataset planning (which webcam scenarios are typical)
- [P] LoRA collection enumeration (which LoRAs to test in Phase 4)

---

## Estimated Wall Time

| Block | Cascade | User | Notes |
|---|---|---|---|
| 0.1 | 1 hr | 5 min | Mostly done |
| 0.2 | — | 30 min | TRT download + extract |
| 0.3 | 4 hr | — | Reading + notes |
| 0.4 | 1 hr | 30 min | User runs Python scripts |
| 0.5 | 4 hr | — | First ONNX export, debug expected |
| 0.6 | 1 hr | — | Mostly trtexec time |
| 0.7 | 6 hr | 1 hr review | First real C++ code |
| 0.8 | 8 hr | 30 min review | Hardest task in Phase 0 |
| 0.9 | 3 hr | 15 min | Refit is mostly mechanical |
| 0.10 | 2 hr | 1 hr review | Writeup + decision |
| **Total** | **30 hr** | **~3 hr** | Spread across 1.5 weeks |
