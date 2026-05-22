"""
Export FLUX.2-Klein-4B model components to ONNX.

This script is a build-time tool only (Constitution Article III).
It runs in the Python environment of the upstream FluxRT repo.

Usage:
    # From a conda env with FluxRT installed
    python tools/export_onnx.py \\
        --models-path ../FluxRT/FLUX.2-klein-4B \\
        --output-dir engines/onnx \\
        --resolution 512 512 \\
        --component transformer

Components:
    transformer  - main 4B FLUX.2 DiT transformer (largest, ~7-8 GB ONNX)
    vae          - autoencoder decoder (small, ~150 MB ONNX)
    text_encoder - Qwen3 text encoder (large, ~3 GB ONNX)

Phase 0 spike: only `transformer` is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_transformer(
    models_path: Path,
    output_path: Path,
    height: int,
    width: int,
    fp16: bool = True,
) -> None:
    """Export the FLUX.2-Klein-4B transformer to ONNX.

    Phase 0 spike: fixed batch=1, fixed resolution. No dynamic shapes.

    Args:
        models_path: Path to FLUX.2-klein-4B/ directory (contains transformer/).
        output_path: Path to write the ONNX file.
        height: Input image height in pixels.
        width: Input image width in pixels.
        fp16: Whether to export in float16 (recommended for TRT input).
    """
    # NOTE: Imports are deferred so this script can be inspected without the
    # heavy ML dependencies installed.
    import torch  # noqa: PLC0415
    from diffusers.models.transformers.transformer_flux2 import (  # noqa: PLC0415
        Flux2Transformer2DModel,
    )

    print(f"[export_onnx] Loading transformer from {models_path}...")
    dtype = torch.float16 if fp16 else torch.float32
    transformer = Flux2Transformer2DModel.from_pretrained(
        str(models_path / "transformer"),
        local_files_only=True,
        torch_dtype=dtype,
    ).eval().cuda()

    # FLUX.2 latent has 16 channels, downsamples 8x from pixel space.
    h_latent = height // 8
    w_latent = width // 8

    # Sequence length for image latent tokens (after patchifying).
    # Patch size is typically 2x2.
    patch_size = 2
    seq_len_img = (h_latent // patch_size) * (w_latent // patch_size)

    # Dummy inputs matching the transformer.forward(...) signature.
    # Exact shapes need to be verified against the FLUX.2-Klein code path.
    # TODO(phase-0): cross-check against transformer_flux2.py forward signature.
    print(
        f"[export_onnx] Building dummy inputs: "
        f"latent={h_latent}x{w_latent}, seq_len_img={seq_len_img}"
    )

    hidden_states = torch.randn(
        1, seq_len_img, 64,  # 16 ch * 2 * 2 patch
        device="cuda", dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        1, 256, 4096,  # text token seq, Qwen3 hidden dim — verify!
        device="cuda", dtype=dtype,
    )
    timestep = torch.tensor([0.5], device="cuda", dtype=dtype)
    img_ids = torch.zeros(seq_len_img, 3, device="cuda", dtype=dtype)
    txt_ids = torch.zeros(256, 3, device="cuda", dtype=dtype)
    guidance = torch.tensor([3.5], device="cuda", dtype=dtype)

    args = (
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        guidance,
    )
    input_names = [
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "img_ids",
        "txt_ids",
        "guidance",
    ]
    output_names = ["sample"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export_onnx] Exporting to {output_path}...")

    with torch.inference_mode():
        torch.onnx.export(
            transformer,
            args,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=20,
            export_params=True,
            do_constant_folding=True,
            external_data=True,  # 4B params won't fit in single .onnx file
        )

    print(f"[export_onnx] Done. Wrote {output_path} (with external data).")


def export_vae(*args, **kwargs):
    raise NotImplementedError("VAE export deferred to Phase 1.")


def export_text_encoder(*args, **kwargs):
    raise NotImplementedError("Text encoder export deferred to Phase 1.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-path", type=Path, required=True,
        help="Path to FLUX.2-klein-4B/ directory."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("engines/onnx"),
        help="Directory to write ONNX files."
    )
    parser.add_argument(
        "--component", choices=["transformer", "vae", "text_encoder"],
        default="transformer",
    )
    parser.add_argument(
        "--resolution", nargs=2, type=int, default=[512, 512],
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--fp32", action="store_true",
                        help="Export in float32 instead of float16.")
    args = parser.parse_args()

    height, width = args.resolution
    fp16 = not args.fp32

    if args.component == "transformer":
        out = args.output_dir / f"flux2_transformer_{height}x{width}_{'fp16' if fp16 else 'fp32'}.onnx"
        export_transformer(args.models_path, out, height, width, fp16=fp16)
    elif args.component == "vae":
        out = args.output_dir / f"flux2_vae_{height}x{width}.onnx"
        export_vae(args.models_path, out, height, width)
    elif args.component == "text_encoder":
        out = args.output_dir / "qwen3_text_encoder.onnx"
        export_text_encoder(args.models_path, out)
    else:
        print(f"unknown component: {args.component}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
