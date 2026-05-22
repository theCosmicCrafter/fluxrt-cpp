# Phase 0 — De-risking Spike

**Duration:** 1.5 weeks
**Goal:** Prove the three highest-risk components work *before* committing
to the full 8-week rewrite. Failures here are cheap; failures in Phase 2
would cost weeks.

---

## What We're De-risking

| # | Risk | What success looks like |
|---|---|---|
| 1 | FLUX.2-Klein transformer can be exported to ONNX | `flux2_transformer.onnx` opens in Netron without errors |
| 2 | TensorRT 10 can compile that ONNX into a `.plan` engine | `trtexec --loadEngine` runs without errors |
| 3 | C++ harness can run the engine and produce output matching Python | PSNR ≥ 40 dB vs Python reference at identical seed |
| 4 | The custom **Spatial KV Cache** can be extracted from `transformer_flux2.py` and reimplemented as a standalone CUDA kernel | Bit-exact output at 0% dynamic area vs Python |
| 5 | TensorRT engine refit (for runtime LoRA hot-swap) actually works on a FLUX engine | Output PSNR ≤ 30 dB vs base after refitting LoRA weights (proves refit took effect) |

---

## Sequence (matters!)

Phase 0 tasks are **mostly sequential**, not parallel, because each step
gates the next:

```
0.1 Repo bootstrap
    │
0.2 Install verification (CUDA + TRT + MSVC)
    │
0.3 Reference repo study (read stable-diffusion.cpp's FLUX path)
    │
0.4 Capture Python reference latents     ◄── needs Python FluxRT working
    │
0.5 Export FLUX.2-Klein transformer to ONNX  ◄── GO/NO-GO #1
    │
0.6 Build TRT engine from ONNX               ◄── GO/NO-GO #2
    │
0.7 C++ engine harness, parity test          ◄── GO/NO-GO #3
    │
0.8 Spatial KV cache CUDA port               ◄── GO/NO-GO #4
    │
0.9 LoRA engine refit spike                  ◄── GO/NO-GO #5
    │
0.10 Phase 0 results writeup, go/no-go decision for Phase 1
```

The only parallelism is during Phase 0.7 — while the engine builds
(can take 10–15 min), I can write the harness scaffolding.

---

## Deliverables

By end of Phase 0:

1. `tools/export_onnx.py` — exports FLUX.2-Klein components to ONNX
2. `tools/build_engine.py` — builds TRT engine from ONNX
3. `tools/capture_reference.py` — captures Python latent fixtures
4. `tests/fixtures/` — 10 (input, prompt, seed) → reference output tuples
5. `src/inference/kernels/spatial_kv_cache.cu` — standalone CUDA kernel
6. `tests/kernels/test_spatial_kv_cache.cpp` — Catch2 test, bit-exact gate
7. A working C++ executable (`fluxrt_spike`) that loads a TRT engine, runs
   one forward pass, and prints PSNR vs reference
8. `specs/phase-0-spike/results.md` — measurements + go/no-go decision

---

## Exit Gates

All five must pass to enter Phase 1:

| Gate | Threshold | Measured by |
|---|---|---|
| Transformer parity | PSNR ≥ 40 dB | `tests/parity/test_transformer_parity.cpp` |
| Spatial cache correctness | bit-exact at 0% dynamic | `tests/kernels/test_spatial_kv_cache.cpp` |
| LoRA refit functional | output PSNR ≤ 30 dB vs base | `tests/parity/test_lora_refit.cpp` |
| Build reproducibility | clean `cmake --preset windows && cmake --build build` | manual |
| Compute-sanitizer clean | no errors on spatial cache test | `compute-sanitizer` invocation |

---

## Failure Plan

If a gate fails:

1. **Don't power through.** Update `BLOCKED.md` with the specific failure.
2. **Document in `results.md`** what was tried and why it failed.
3. **Escalate paths:**
   - Gate 1 (transformer parity): Try Torch-TensorRT in-process compile as
     a hybrid fallback. Loses the "no Python at runtime" goal but keeps
     performance.
   - Gate 2 (spatial cache): The Python implementation patches diffusers
     deeply; consider keeping the cache logic in PyTorch via `torch.compile`
     and only porting the inner attention kernel to CUDA.
   - Gate 3 (LoRA refit): Drop runtime hot-swap from v1, switch to
     pre-baked LoRAs (offline merge). User pre-approved this fallback.
4. **User review.** Halt and discuss before proceeding.

---

## Reference Material

In `c:/Users/richk/refs/`:

- `stable-diffusion.cpp` — leejet's pure C++ FLUX inference. Study their
  scheduler, LoRA loader, and tokenizer. **Best single reference.**
- `NVIDIA-TensorRT/demo/Diffusion` — production diffusion pipeline with
  refit + LoRA. Direct pattern for our Phase 0.9.
- `Torch-TensorRT/examples/apps/flux_demo.py` — Python reference for
  exporting FLUX to TensorRT. Adapt to FLUX.2-Klein.
- `Model-Optimizer` — needed for Phase 5; skim Phase 0.
- `StreamDiffusion` — real-time streaming patterns. Skim only.
- `Practical-RIFE` — needed for Phase 3; skim Phase 0 to find an existing
  C++ port if any.

---

## See Also

- `tasks.md` — actionable task list with `[P]` parallel markers
- `results.md` — written at end of Phase 0 with measurements + decision
- `../../CONSTITUTION.md` — non-negotiable rules
- `../../BLOCKED.md` — items waiting on user action
