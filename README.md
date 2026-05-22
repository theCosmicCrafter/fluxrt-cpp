# FluxRT-CPP

A C++ port of [FluxRT](https://github.com/tensorforger/FluxRT) — real-time
FLUX.2-Klein stream editing — built on TensorRT for maximum performance on
NVIDIA RTX 4090 / 5090.

**Repo:** https://github.com/theCosmicCrafter/fluxrt-cpp

**Target model:** [FLUX.2-Klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(base, FP16 single-file `flux-2-klein-4b.safetensors`, ~7.2 GB).

**Phase 5+ optimization candidates:**
- [FLUX.2-Klein-AIO](https://civitai.com/models/2327389/flux2-klein-aio) —
  4–6-step distilled, Apache-2.0. Could yield 5× speedup once we're confident
  the C++ pipeline matches base Klein numerics.
- NVFP4 quantization via NVIDIA Model-Optimizer (Blackwell-native, hardware-
  accelerated FP4 tensor cores).

> **Status:** Phase 0 (de-risking spike). Not yet usable. See `specs/` for
> phase plans and `BLOCKED.md` for current dependencies.

## Goals

- **2× FPS** vs upstream Python FluxRT on RTX 5090
- **~50% VRAM reduction** via NVFP4 (Blackwell) quantization
- **Single binary distribution** — no Python or conda at runtime
- **Runtime LoRA hot-swap** via TensorRT engine refit
- **Cross-OS:** Windows 11 + Linux x86_64, both NVIDIA CUDA

## Non-Goals (v1)

- AMD / Intel / Apple Silicon support
- CPU-only fallback
- Browser / WASM target

## Tech Stack

| Layer | Choice |
|---|---|
| Inference | NVIDIA TensorRT 10.16+ |
| Compiler | MSVC 19.40+ (Win) / GCC 13+ (Linux) |
| Build | CMake 3.28+ with vcpkg manifest |
| GPU | CUDA 12.8 or 13.x |
| GUI | SDL3 + Dear ImGui |
| Tests | Catch2 + compute-sanitizer |
| Quant | NVIDIA Model-Optimizer (NVFP4 / FP8 / INT8) |

## Quick Status

- **Constitution:** `CONSTITUTION.md` — non-negotiable rules
- **Blocked items:** `BLOCKED.md` — what's waiting on user action
- **Current phase:** `specs/phase-0-spike/`

## Required Tools

See `BLOCKED.md` for setup status. At minimum:

- NVIDIA RTX 4090 or 5090
- CUDA Toolkit 12.8 or 13.x
- TensorRT 10.16+ (download from NVIDIA developer portal)
- Visual Studio 2022 (Community or Build Tools) with C++ workload, OR
  GCC 13+ on Linux
- CMake 3.28+
- Python 3.12 (for ONNX export tooling — build-time only)

## Development Workflow

1. Each phase begins with `specs/phase-N/plan.md` and `tasks.md`
2. Cascade (the AI agent) executes tasks; user reviews at gates
3. Manual benchmark gate at every phase merge (no CI in v1)
4. Phase ends with `specs/phase-N/results.md` capturing measurements
5. Constitution review at end of every phase

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Vendored third-party code lives in
`third_party/` and may have compatible permissive licenses (see
`LICENSES.md` once that file exists).

## Acknowledgements

- Upstream Python [FluxRT](https://github.com/tensorforger/FluxRT) by tensorforger
- [Black Forest Labs](https://bfl.ai/) for FLUX.2-Klein
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) by leejet — reference C++ implementation patterns
- NVIDIA TensorRT and Model-Optimizer teams
