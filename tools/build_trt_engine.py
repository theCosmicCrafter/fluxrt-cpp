"""Phase 0.6: Build TensorRT engine from ONNX.

Usage:
    # Set PATH so tensorrt can find nvinfer_10.dll
    $env:PATH = "C:\\Users\\richk\\CascadeProjects\\TensorRT-10.16.1.11\\bin;$env:PATH"
    .\venv\Scripts\python tools\build_trt_engine.py
"""
import argparse
import os
import sys
import time
from pathlib import Path

# TensorRT lib must be on PATH before import
def _ensure_trt_path():
    trt_bin = Path("C:/Users/richk/CascadeProjects/TensorRT-10.16.1.11/bin")
    if trt_bin.exists():
        os.environ["PATH"] = str(trt_bin) + os.pathsep + os.environ.get("PATH", "")

_ensure_trt_path()

try:
    import tensorrt as trt
except Exception as e:
    print(f"ERROR: Cannot import tensorrt: {e}")
    print("Ensure TensorRT 10.x bin/ is on PATH.")
    sys.exit(1)

import onnx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ONNX_PATH = Path("engines/onnx/flux_2_klein_4b_transformer_512x512.onnx")
ENGINE_DIR = Path("engines")
DEFAULT_ENGINE_NAME = "flux_2_klein_4b_transformer.plan"

# Shape profiles: (min, opt, max)
# For 512x512 image -> latent 64x64 = 4096 tokens (Flux2 VAE scale factor 8 + patch 2)
# For 1024x1024 image -> latent 128x128 = 16384 tokens
BATCH_PROFILE = (1, 1, 2)
IMG_SEQ_PROFILE = (4096, 4096, 4096)   # Fixed: 64x64 latent for 512px image
TXT_SEQ_PROFILE = (256, 256, 256)        # Fixed: 256 text tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_engine(
    onnx_path: Path,
    output_path: Path,
    fp16: bool = True,
    bf16: bool = False,
    workspace_mb: int = 8192,
) -> Path:
    """Build a TensorRT engine from an ONNX model."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX (path-based so external data files are resolved)
    print(f"[trt] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error {i}: {parser.get_error(i)}")
        raise RuntimeError("ONNX parsing failed")
    print(f"  {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] Enabled FP16")
    if bf16 and hasattr(trt.BuilderFlag, "BF16"):
        config.set_flag(trt.BuilderFlag.BF16)
        print("[trt] Enabled BF16")

    # Profile (must match ONNX input names exactly)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "hidden_states",
        min=(BATCH_PROFILE[0], IMG_SEQ_PROFILE[0], 128),
        opt=(BATCH_PROFILE[1], IMG_SEQ_PROFILE[1], 128),
        max=(BATCH_PROFILE[2], IMG_SEQ_PROFILE[2], 128),
    )
    profile.set_shape(
        "encoder_hidden_states",
        min=(BATCH_PROFILE[0], TXT_SEQ_PROFILE[0], 7680),
        opt=(BATCH_PROFILE[1], TXT_SEQ_PROFILE[1], 7680),
        max=(BATCH_PROFILE[2], TXT_SEQ_PROFILE[2], 7680),
    )
    profile.set_shape(
        "timestep",
        min=(BATCH_PROFILE[0],),
        opt=(BATCH_PROFILE[1],),
        max=(BATCH_PROFILE[2],),
    )
    profile.set_shape(
        "img_ids",
        min=(BATCH_PROFILE[0], IMG_SEQ_PROFILE[0], 4),
        opt=(BATCH_PROFILE[1], IMG_SEQ_PROFILE[1], 4),
        max=(BATCH_PROFILE[2], IMG_SEQ_PROFILE[2], 4),
    )
    profile.set_shape(
        "txt_ids",
        min=(BATCH_PROFILE[0], TXT_SEQ_PROFILE[0], 4),
        opt=(BATCH_PROFILE[1], TXT_SEQ_PROFILE[1], 4),
        max=(BATCH_PROFILE[2], TXT_SEQ_PROFILE[2], 4),
    )
    config.add_optimization_profile(profile)
    print(f"[trt] Optimization profile:")
    print(f"  batch:        {BATCH_PROFILE}")
    print(f"  img_seq_len:  {IMG_SEQ_PROFILE}")
    print(f"  txt_seq_len:  {TXT_SEQ_PROFILE}")

    # Build
    print(f"[trt] Building engine (this may take several minutes)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed — see TensorRT logs above")
    dt = time.time() - t0
    print(f"[trt] Build completed in {dt:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[trt] Serialized engine: {output_path} ({size_mb:.1f} MB)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH)
    parser.add_argument("--output", type=Path, default=ENGINE_DIR / DEFAULT_ENGINE_NAME)
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 (default)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable BF16")
    parser.add_argument("--workspace", type=int, default=8192, help="Workspace in MB")
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"ERROR: ONNX model not found: {args.onnx}")
        return 1

    try:
        build_engine(
            onnx_path=args.onnx,
            output_path=args.output,
            fp16=args.fp16,
            bf16=args.bf16,
            workspace_mb=args.workspace,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("\nOK Phase 0.6 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
