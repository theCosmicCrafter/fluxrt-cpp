# BLOCKED — Items Awaiting User Action

This file tracks anything Cascade cannot do alone. Updated proactively.

**Priority:** P0 = blocks current phase, P1 = blocks next phase, P2 = nice-to-have.

---

## Active Blockers

### P0 — Required to start Phase 0 work

- [ ] **Hugging Face authentication** *(date: 2026-05-22)*
      Action needed:
      1. Get an HF access token (read scope) at
         https://huggingface.co/settings/tokens
      2. Run `huggingface-cli login` and paste the token.
      3. Accept the FLUX.2-Klein license on
         https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
      Workaround: Cascade can clone via git if you provide creds, but
      `huggingface-cli` is the cleanest path.

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
