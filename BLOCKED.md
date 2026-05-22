# BLOCKED — Items Awaiting User Action

This file tracks anything Cascade cannot do alone. Updated proactively.

**Priority:** P0 = blocks current phase, P1 = blocks next phase, P2 = nice-to-have.

---

## Active Blockers

### P0 — Required to start Phase 0 work

- [ ] **TensorRT 10 download & install** *(date: 2026-05-22)*
      Action needed:
      1. Sign in / register at https://developer.nvidia.com/tensorrt
      2. Download TensorRT 10.7 or newer for Windows, CUDA 13.x, zip package
      3. Extract to `C:\TensorRT-10.x` (or your chosen path)
      4. Add `C:\TensorRT-10.x\bin` to PATH
      5. Add `C:\TensorRT-10.x\lib` to PATH
      6. Verify with: `trtexec --help`
      Workaround: None. NVIDIA gates this download behind their EULA, no
      automation possible.

- [ ] **GitHub repo creation** *(date: 2026-05-22)*
      Action needed: Create empty private repo named `FluxRT-CPP` under your
      GitHub account. Do not initialize with README/license/gitignore — we
      have those locally already.
      Workaround: None.

- [ ] **Hugging Face authentication** *(date: 2026-05-22)*
      Action needed:
      1. Get an HF access token (read scope) at
         https://huggingface.co/settings/tokens
      2. Run `huggingface-cli login` and paste the token.
      3. Accept the FLUX.2-Klein license on
         https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
      Workaround: Cascade can clone via git if you provide creds, but
      `huggingface-cli` is the cleanest path.

### P1 — Required during Phase 0

- [ ] **CUDA 13.2 install completion + reboot** *(date: 2026-05-22)*
      Status: winget install kicked off in background.
      Action needed: Approve UAC if prompted. Reboot if installer asks.
      Verification command: `nvcc --version`

- [ ] **Visual Studio 2022 (Community + C++ workload) install** *(date: 2026-05-22)*
      Status: Will start once CUDA install completes.
      Action needed: Approve UAC if prompted.

### P2 — Required before Phase 5 (Quantization)

- [ ] **Calibration dataset (~1000 representative input frames)** *(date: TBD)*
      Action needed: Either provide a folder of typical input frames or grant
      Cascade webcam access via a capture script to record them.

---

## Resolved Blockers

*(Items get moved here once unblocked. Empty for now.)*

---

*Updated by Cascade automatically. Last edit: 2026-05-22 13:15 ET.*
