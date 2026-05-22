# Phase 0 — De-risking Spike

**Duration:** 1.5 weeks
**Goal:** Prove the three highest-risk components work *before* committing
to the full 8-week rewrite. Failures here are cheap; failures in Phase 2
would cost weeks.

---

## What We're De-risking

| # | Risk | What success looks like |
|---|---|---|
| 1 | **AIO single-file format loads via diffusers `from_single_file()`** and decomposes into transformer / VAE / text encoder | All three component objects accessible; can `forward()` each one independently |
| 2 | **Spatial-cache patches in `transformer_flux2.py` apply to AIO architecture** (AIO is distilled but should be structurally identical) | FluxRT Python pipeline runs end-to-end with AIO weights at 4–6 steps |
| 3 | AIO transformer can be exported to ONNX | `flux2_transformer_aio.onnx` opens in Netron without errors |
| 4 | TensorRT 10 can compile that ONNX into a `.plan` engine | `trtexec --loadEngine` runs without errors |
| 5 | C++ harness can run the engine and produce output matching Python AIO reference | PSNR ≥ 40 dB vs Python AIO reference at identical seed |
| 6 | The custom **Spatial KV Cache** ports as a standalone CUDA kernel | Bit-exact output at 0% dynamic area vs Python |
| 7 | TensorRT engine refit (for runtime LoRA hot-swap) works on a FLUX-distilled engine | Output PSNR ≤ 30 dB vs base after refitting LoRA weights |

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
0.4a Download AIO weights from CivitAI       ◄── needs civitai API key
0.4b Verify AIO loads in upstream FluxRT     ◄── GO/NO-GO #1 (architecture)
0.4c Capture AIO reference latents (10 fixtures)
    │
0.5 Export AIO transformer to ONNX           ◄── GO/NO-GO #2 (export)
    │
0.6 Build TRT engine from ONNX               ◄── GO/NO-GO #3 (compile)
    │
0.7 C++ engine harness, parity test          ◄── GO/NO-GO #4 (numerics)
    │
0.8 Spatial KV cache CUDA port               ◄── GO/NO-GO #5 (kernel)
    │
0.9 LoRA engine refit spike                  ◄── GO/NO-GO #6 (refit)
    │
0.10 Phase 0 results writeup, go/no-go decision for Phase 1
```

### Special Note on AIO Distillation

AIO is a 4–6-step distilled FLUX.2-Klein-4B. Implications:

- **Scheduler config differs.** Distilled models use sigma schedules tuned for
  the smaller step count. We must capture the AIO scheduler config alongside
  the weights and reproduce it in the C++ runtime exactly.
- **Architecture should be identical** to base Klein-4B (distillation only
  changes weights, not topology). If it isn't, the FluxRT spatial-cache
  patches in `transformer_flux2.py` will break and we have a much bigger
  problem. Risk #2 above tests this.
- **Real-time math improves dramatically.** At 6 steps + ~100ms per step on
  RTX 5090, that's 600ms per generation — within reach of streaming use cases
  with frame-level caching from FluxRT's spatial cache.

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
