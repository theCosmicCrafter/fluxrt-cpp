# BLOCKED — Items Awaiting User Action

This file tracks anything Cascade cannot do alone. Updated proactively.

**Priority:** P0 = blocks current phase, P1 = blocks next phase, P2 = nice-to-have.

---

## Active Blockers

### P0 — Required to start Phase 0 work

- [x] **Hugging Face configs for FLUX.2-klein-4B** *(resolved: 2026-05-22)*
      The base Klein-4B repo is **public** (Apache-2.0). Downloaded all
      configs + tokenizer without auth via huggingface-hub Python API:
      `snapshot_download(..., allow_patterns=["*.json", "tokenizer*"])`.
      Result: 19 files in `models/base-config/` including `model_index.json`,
      `scheduler/scheduler_config.json`, tokenizer vocab files, and
      component configs for transformer, vae, and text_encoder.

### P2 — Required before Phase 5 (Quantization)

- [ ] **Calibration dataset (~1000 representative input frames)** *(date: TBD)*
      Action needed: Either provide a folder of typical input frames or grant
      Cascade webcam access via a capture script to record them.

---

## Resolved Blockers

- [x] **CUDA Toolkit installed** *(resolved: 2026-05-22)*
      CUDA 13.2.78 installed via `winget install Nvidia.CUDA`.
      Verified: `nvcc --version` returns `release 13.2, V13.2.78`.

- [x] **Visual Studio 2022 Build Tools installed** *(resolved: 2026-05-22)*
      Installed via `winget install Microsoft.VisualStudio.2022.BuildTools`
      with NativeDesktop / VCTools / Windows11SDK / CMake workloads.
      MSVC 14.44.35207 at standard path. `cl.exe` and `vcvars64.bat`
      both present.

- [x] **TensorRT 10 download & install** *(resolved: 2026-05-22)*
      TensorRT 10.16.1.11 extracted to
      `C:\Users\richk\CascadeProjects\TensorRT-10.16.1.11`.
      `bin/`, `lib/`, `include/` populated. `trtexec.exe` present.
      User PATH and `TENSORRT_ROOT` env var set.

- [x] **GitHub repo creation + initial push** *(resolved: 2026-05-22)*
      Repo at https://github.com/theCosmicCrafter/fluxrt-cpp.
      Initial commit `361c13a6` pushed to `main`.

- [x] **Base FLUX.2-Klein-4B weights** *(resolved: 2026-05-22)*
      User provided `flux-2-klein-4b.safetensors` (7.22 GB FP16) directly.
      Now at `models/base/flux-2-klein-4b.safetensors`.

- [x] **Brief AIO pivot, then back to base Klein** *(resolved: 2026-05-22)*
      Briefly considered FLUX.2-klein-AIO (4–6 step distilled, Apache 2.0)
      and SDNQ-4bit-dynamic variants. Decided to start with base Klein-4B
      for clean precision. AIO + NVFP4 documented as Phase 5+ optimizations
      in `specs/phase-0-spike/plan.md`.

---

*Updated by Cascade automatically. Last edit: 2026-05-22 13:50 ET.*
