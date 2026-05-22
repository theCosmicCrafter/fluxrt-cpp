# Phase 0.3 Reference Repo Study

Date: 2026-05-22

## 1. stable-diffusion.cpp (GGML-based FLUX inference)

**File:** `src/flux.hpp`

### Architecture Mapping

The GGML implementation confirms the FLUX transformer structure:

- **DoubleStreamBlock** (19x in FLUX.1-dev, **5x in Klein-4B**):
  - Parallel image + text streams
  - Each stream: RMSNorm → SelfAttention (QKNorm + RoPE) → MLP
  - Shared attention: concat(img_q, txt_q) across sequence dim
  - Modulation: shift + scale + gate from timestep embedding

- **SingleStreamBlock** (38x in FLUX.1-dev, **20x in Klein-4B**):
  - Fused QKV+MLP projection (`linear1`) then split
  - Attention output concat with MLP output → `linear2`
  - Single modulation (not double)

- **SelfAttention**:
  - QKNorm per-head (RMSNorm on query/key before RoPE)
  - RoPE positional embeddings via `Rope::attention()`
  - No traditional causal mask (FLUX uses full attention)

- **MLP variants**:
  - Standard: GELU(approximate="tanh")
  - `use_mlp_silu_act`: SiLU activation (used in FLUX2/Klein)
  - `YakMLP`: Gate-style MLP (gate_proj + up_proj + down_proj)

### Key C++ Patterns for FluxRT

1. **Graph building**: `ggml_cgraph` constructed once, then `ggml_graph_compute()`
2. **Backend abstraction**: `ggml_backend_t` for CPU/CUDA/Metal
3. **Tensor storage map**: `String2TensorStorage` maps weight names to buffers
4. **Weight loading**: `model_loader.load_tensors(tensors)` populates graph params
5. **Memory management**: `alloc_params_buffer()` / `free_params_buffer()`

### FLUX2-specific flags (match Klein-4B config)

From `FluxRunner` constructor and `FluxParams`:
- `in_channels = 128`, `patch_size = 1`
- `mlp_ratio = 3.0f`
- `theta = 2000`
- `axes_dim = {32, 32, 32, 32}` (4D RoPE, not 3D)
- `vec_in_dim = 0` (no pooled CLIP embeddings — Qwen3 only)
- `qkv_bias = false`, `disable_bias = true`
- `share_modulation = true` (shared across blocks)
- `use_mlp_silu_act = true`

---

## 2. Torch-TensorRT (flux_demo.py)

**Approach:** `torch_tensorrt.MutableTorchTensorRTModule` — JIT compilation of PyTorch modules to TRT, NOT ONNX export.

### Relevant for FluxRT:

- **Quantization**: Uses `modelopt.torch.quantization` (NVFP4, FP8, INT8)
  - Configs: `NVFP4_DEFAULT_CFG`, `FP8_DEFAULT_CFG`, `INT8_DEFAULT_CFG`
  - Calibration via forward loop on real prompts
  - Filter func disables quantizers on embedders/norms

- **Dynamic shapes**: `torch.export.Dim("batch", min=1, max=8)`
  - Only batch dim is dynamic; image resolution is fixed at export time

- **LoRA refit pattern** (lines 179-189):
  ```python
  pipeline.load_lora_weights(path, adapter_name="lora1")
  pipeline.set_adapters(["lora1"], adapter_weights=[1])
  pipeline.fuse_lora()
  pipeline.unload_lora_weights()
  # Then run inference — MutableTorchTensorRTModule auto-refits
  ```
  This is PyTorch-side LoRA fusion, not TRT engine refit.

### What we DON'T take:
- `MutableTorchTensorRTModule` is PyTorch-only; we need ONNX → TRT for C++
- `from_pretrained()` loading (we use `from_single_file()` for local safetensors)

---

## 3. NVIDIA TensorRT (engine_refit samples)

### Two refit APIs:

**A. Manual weight refit (`build_and_refit_engine.py`)**
```python
config.set_flag(trt.BuilderFlag.REFIT)  # at build time
refitter = trt.Refitter(engine, logger)
refitter.weights_validation = False
refitter.set_named_weights(name, trt.Weights(...), trt.TensorLocation.DEVICE)
assert refitter.refit_cuda_engine()
```
- Must build engine with `REFIT` flag
- Can refit from GPU or CPU memory
- `get_missing_weights()` validates completeness

**B. ONNX parser refit (`refit_engine_and_infer.py`)**
```python
refitter = trt.Refitter(engine, logger)
parser_refitter = trt.OnnxParserRefitter(refitter, logger)
assert parser_refitter.refit_from_file(onnx_path)
assert refitter.refit_cuda_engine()
```
- Uses ONNX file as weight source
- Simpler for LoRA: export fused model to ONNX, then refit engine

### For FluxRT LoRA (Phase 6):
- Build base engine with `REFIT` flag
- Fuse LoRA in PyTorch → export to ONNX → `OnnxParserRefitter`
- Or manually map LoRA weight deltas to `set_named_weights`

---

## 4. Architecture Comparison Table

| Component | FLUX.1-dev | FLUX.2-Klein-4B | Notes |
|-----------|-----------|-----------------|-------|
| Double blocks | 19 | **5** | Joint img/txt attention |
| Single blocks | 38 | **20** | Fused self-attn + MLP |
| Hidden size | 3072 | 3072 | |
| Attention heads | 24 | **24** | head_dim = 128 |
| MLP ratio | 4.0 | **3.0** | SiLU activation |
| Patch size | 2 | **1** | No patching needed |
| In channels | 64 | **128** | Latent channels |
| RoPE theta | 10000 | **2000** | 4D axes {32,32,32,32} |
| Text encoder | CLIP-L + T5-XXL | **Qwen3** | 7680-dim joint dim |
| Pooled embed | 768-dim | **None** (`vec_in_dim=0`) | |
| Guidance embed | Yes | **No** | `guidance_embeds=false` |
| Scheduler shift | 1.0-1.15 | **3.0** | Dynamic shifting |
| Distilled | No | **Yes** | Fewer steps |

---

## 5. Implications for ONNX Export

1. **Smaller model** = faster export and engine build
2. **Qwen3 text encoder** = different from standard FLUX.1-dev export tutorials
3. **No guidance embedding** = one fewer input tensor
4. **No pooled embeddings** = `pooled_projections` input absent
5. **Patch size 1, in_channels 128** = latent shape is `[B, 128, H, W]` not `[B, 16, H/8, W/8]`
   - Wait: FLUX VAE still outputs 16-channel latents. The `in_channels=128` on transformer means it operates on a different latent representation. Need to verify.
6. **Shared modulation** = modulation weights may be shared across blocks in the checkpoint

### Open Questions
- Does `in_channels=128` mean the VAE outputs 128 channels? Or is there a projection?
- The `flux-2-klein-4b.safetensors` has a `model.diffusion_model.` prefix or diffusers format?
- Can `diffusers.Flux2KleinPipeline.from_pretrained()` load from our local config + weights?

---

## Next Steps

Phase 0.4: Load the model in Python, verify architecture matches config, capture reference fixtures for parity testing.
