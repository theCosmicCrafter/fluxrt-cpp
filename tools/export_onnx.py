"""
Export FLUX.2-Klein transformer to ONNX for TensorRT engine build.

This script is a build-time tool only (Constitution Article III).

Usage:
    venv/Scripts/python tools/export_onnx.py \\
        --checkpoint models/base/flux-2-klein-4b.safetensors \\
        --output-dir engines/onnx \\
        --height 512 --width 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
from safetensors.torch import load_file


class ManualRMSNorm(nn.Module):
    """ONNX-compatible RMSNorm using standard ops."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def replace_rmsnorm(module):
    """Recursively replace nn.RMSNorm with ManualRMSNorm in place."""
    for name, child in module.named_children():
        if isinstance(child, nn.RMSNorm):
            # Get config from original module
            dim = child.weight.shape[0]
            eps = child.eps
            new_norm = ManualRMSNorm(dim, eps=eps)
            # Copy weight
            with torch.no_grad():
                new_norm.weight.copy_(child.weight)
            setattr(module, name, new_norm)
        else:
            replace_rmsnorm(child)


def load_transformer(checkpoint: Path, config_path: Path):
    """Load transformer from safetensors + config."""
    print(f"[export] Loading config: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print(f"[export] Loading weights: {checkpoint}")
    state_dict = load_file(str(checkpoint), device="cpu")
    print(f"  Keys in checkpoint: {len(state_dict)}")

    print("[export] Converting checkpoint keys to diffusers format...")
    state_dict = convert_flux2_transformer_checkpoint_to_diffusers(state_dict)
    print(f"  Converted keys: {len(state_dict)}")

    print("[export] Building model...")
    model = Flux2Transformer2DModel.from_config(config)
    model.load_state_dict(state_dict, strict=True)
    print("  State dict loaded (strict=True) — all keys matched!")
    return model


def build_dummy_inputs(device: str, batch_size: int, height: int, width: int,
                       txt_seq_len: int, joint_attention_dim: int,
                       in_channels: int, guidance_embeds: bool):
    """Build dummy inputs matching the verified Phase 0.4 shapes."""
    img_seq_len = height * width

    # hidden_states: packed [B, H*W, C]
    latents_4d = torch.randn(
        batch_size, in_channels, height, width, dtype=torch.float32, device=device,
    )
    hidden_states = latents_4d.reshape(batch_size, in_channels, img_seq_len).permute(0, 2, 1)

    # img_ids: 4D coords (T, H, W, L)
    img_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1),
    ).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    # encoder_hidden_states
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim,
                                        dtype=torch.float32, device=device)

    # txt_ids
    txt_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(txt_seq_len),
    ).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    timestep = torch.tensor([0.5], dtype=torch.float32, device=device)
    guidance = torch.tensor([3.0], dtype=torch.float32, device=device) if guidance_embeds else None

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "guidance": guidance,
    }


def export_transformer(
    checkpoint: Path,
    config_path: Path,
    output_path: Path,
    height: int,
    width: int,
    batch_size: int = 1,
    fixed_shapes: bool = False,
) -> None:
    """Export the FLUX.2-Klein transformer to ONNX with dynamic axes."""
    device = "cpu"  # CPU export; RTX 5090 sm_120 not supported by current torch
    model = load_transformer(checkpoint, config_path)
    model.to(device).eval()

    print("[export] Replacing nn.RMSNorm with ONNX-compatible ManualRMSNorm...")
    replace_rmsnorm(model)
    print("  Done.")

    cfg = model.config
    inputs = build_dummy_inputs(
        device=device, batch_size=batch_size, height=height, width=width,
        txt_seq_len=256, joint_attention_dim=cfg.joint_attention_dim,
        in_channels=cfg.in_channels, guidance_embeds=cfg.guidance_embeds,
    )

    # Filter None guidance for models that don't use it
    args = tuple(v for v in inputs.values() if v is not None)
    input_names = [k for k, v in inputs.items() if v is not None]

    # Dynamic axes for batch, image sequence, and text sequence
    if fixed_shapes:
        # NVIDIA-style fixed-sequence export (avoids shape-dependent reshape issues)
        dynamic_axes = {
            "hidden_states": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "timestep": {0: "batch"},
            "img_ids": {0: "batch"},
            "txt_ids": {0: "batch"},
        }
        print("\n[export] Fixed-shape export (seq_len not dynamic) — TensorRT compatible")
    else:
        dynamic_axes = {
            "hidden_states": {0: "batch", 1: "img_seq_len"},
            "encoder_hidden_states": {0: "batch", 1: "txt_seq_len"},
            "timestep": {0: "batch"},
            "img_ids": {0: "batch", 1: "img_seq_len"},
            "txt_ids": {0: "batch", 1: "txt_seq_len"},
        }
    if cfg.guidance_embeds:
        dynamic_axes["guidance"] = {0: "batch"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[export] Exporting to {output_path}...")
    print(f"  opset=17, dynamic_axes={list(dynamic_axes.keys())}")

    with torch.inference_mode():
        print("[export] Trying dynamo_export (torch.export-based)...")
        try:
            onnx_program = torch.onnx.export(
                model,
                args,
                str(output_path),
                input_names=input_names,
                output_names=["sample"],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                export_params=True,
                do_constant_folding=True,
                dynamo=True,
            )
            if onnx_program is not None:
                onnx_program.save(str(output_path))
            print(f"[export] dynamo_export succeeded: {output_path}")
        except Exception as e:
            print(f"[export] dynamo_export failed: {e}")
            print("[export] Falling back to trace-based torch.onnx.export...")
            torch.onnx.export(
                model,
                args,
                str(output_path),
                input_names=input_names,
                output_names=["sample"],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                export_params=True,
                do_constant_folding=True,
                use_external_data_format=True,
            )

    print(f"[export] Done. Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/base/flux-2-klein-4b.safetensors"))
    parser.add_argument("--config", type=Path,
                        default=Path("models/base-config/transformer/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("engines/onnx"))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fixed-shapes", action="store_true",
                        help="Disable dynamic axes (needed for TensorRT build)")
    args = parser.parse_args()

    # Derive latent spatial dimensions from pixel dimensions.
    # The model works on packed latents; height/width here are the spatial
    # dimensions of the packed latent grid (e.g. 64x64 for 512px image).
    # VAE does 8x compression: 512/8 = 64.
    latent_h = args.height // 8
    latent_w = args.width // 8

    stem = args.checkpoint.stem.replace("-", "_").replace(".", "_")
    out = args.output_dir / f"{stem}_transformer_{args.height}x{args.width}.onnx"
    export_transformer(
        args.checkpoint, args.config, out,
        height=latent_h, width=latent_w, batch_size=args.batch_size,
        fixed_shapes=args.fixed_shapes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
