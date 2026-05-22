# FluxRT-CPP Constitution

These principles are **non-negotiable**. Every PR, every kernel, every commit
must comply. Violations are bugs and must be fixed before merge.

---

## Article I — Parity Before Optimization

No optimization may ship until parity tests pass against the Python FluxRT
reference implementation:

- **PSNR ≥ 35 dB** (image space, vs Python output at identical seed)
- **SSIM ≥ 0.97**
- **LPIPS ≤ 0.05**

These thresholds may be relaxed for quantized engines (FP8, NVFP4) only with
explicit per-engine documented exceptions in `specs/phase-N/results.md`.

## Article II — No Regression

Every phase merge must run `tools/run_benchmark.sh` and write results to
`specs/phase-N/results.md`. Compared against the previous phase's results:

- **FPS regression > 5%** blocks merge.
- **VRAM increase > 10%** blocks merge.
- **End-to-end latency increase > 10%** blocks merge.

Exceptions require explicit user approval recorded in the phase results file.

## Article III — CUDA-Only, Single Binary

- No Python is allowed at runtime. Python exists only as a build-time tool
  for ONNX export and engine calibration (in `tools/`).
- The release artifact is a single executable plus the `engines/` directory.
- No `pip install` or `conda activate` is required to run FluxRT-CPP.
- Optional features that require Python (e.g., LivePortrait) must be feature-
  flagged off by default and documented as optional Python subprocess deps.

## Article IV — Deterministic Builds

- `vcpkg.json` is in **manifest mode** with explicit version pins.
- `vcpkg-configuration.json` pins the registry baseline.
- Every dependency is reproducible across machines.
- TensorRT, CUDA Toolkit, MSVC version are all documented in `README.md`
  with exact versions tested.

## Article V — Fail Loudly

- Every CUDA call must be wrapped in `CUDA_CHECK(...)`.
- Every TensorRT call must be wrapped in `TRT_CHECK(...)`.
- No silent fallbacks. If something is unsupported, throw with a clear message.
- All kernel launches checked with `cudaPeekAtLastError()` or
  `cudaGetLastError()` after launch.

## Article VI — One Owner Per Subsystem

Subsystems are isolated. Cross-cutting hacks are forbidden:

| Subsystem | Owner module | May NOT depend on |
|---|---|---|
| Inference | `src/inference/` | GUI, IO |
| Pipeline | `src/pipeline/` | GUI, IO |
| Runtime | `src/runtime/` | Inference, GUI, IO |
| GUI | `src/app/gui.cpp` | Inference internals (only public API) |
| IO | `src/io/` | Inference, GUI |
| Utils | `src/utils/` | Anything (this is the leaf) |

## Article VII — Test First for Kernels

Every `.cu` file in `src/inference/kernels/` and `src/interpolation/` must
have a corresponding test in `tests/kernels/` that:

1. Has at least one bit-exact correctness check vs reference.
2. Runs cleanly under `compute-sanitizer --tool memcheck`.
3. Runs cleanly under `compute-sanitizer --tool racecheck`.
4. Is invoked by `tools/run_benchmark.sh` before benchmarking.

## Article VIII — BLOCKED.md Is Sacred

Anything blocking progress that requires user action is recorded in
`BLOCKED.md` at the repo root. Items are never silently abandoned. Format:

```
- [ ] [P0/P1/P2] <date> <description>
      Action needed: <what user must do>
      Workaround: <if any>
```

The agent (Cascade) updates this file proactively when blocked.

## Article IX — License Boundary

This project is Apache-2.0 licensed. Vendored third-party code in
`third_party/` must have compatible licenses (Apache-2.0, MIT, BSD-3-Clause,
zlib, or public domain). GPL/LGPL code may be linked dynamically only and
must be documented in `LICENSES.md`.

## Article X — Constitution Review

This document is reviewed at the end of every phase. Amendments require:

1. Documented rationale in the phase's `results.md`.
2. Explicit user sign-off.
3. Updated date below.

---

**Ratified:** Phase 0 kickoff
**Last reviewed:** Phase 0 kickoff
