# BLOCKED — Items Awaiting User Action

This file tracks anything Cascade cannot do alone. Updated proactively.

**Priority:** P0 = blocks current phase, P1 = blocks next phase, P2 = nice-to-have.

---

## Active Blockers

### P0 — Required to start Phase 0 work

- [ ] **CivitAI account + API key for AIO model download** *(date: 2026-05-22)*
      Status: Required after pivot to FLUX.2-klein-AIO (a CivitAI-only model).
      Action needed:
      1. Create / sign in at https://civitai.com (will redirect to civitai.red
         since AIO is on the mature-content domain).
      2. Generate an API key at
         https://civitai.com/user/account (API Keys section).
      3. Download `flux2-klein-aio` (single .safetensors, ~16 GB BF16 expected
         for v1, or 7.9 GB for FP8 distill variant) from
         https://civitai.com/models/2327389 to `models/aio/`.
      4. Provide the API key to Cascade so `tools/download_aio.py` can pull
         updates if the upstream version changes.
      Workaround: Manual browser download is fine for the one-time grab.

- [ ] **Hugging Face authentication (still needed for diffusers code)**
      *(date: 2026-05-22)*
      Action needed: `huggingface-cli login` with a read-scope token. We pull
      the FLUX.2-Klein **diffusers code** (Python module) from HF even though
      the AIO weights come from CivitAI. The base model on HF is also useful
      as a known-good reference for parity tests.

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

---

*Updated by Cascade automatically. Last edit: 2026-05-22 13:50 ET.*
