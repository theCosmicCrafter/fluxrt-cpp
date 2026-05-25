"""Phase 0.5: Export FLUX.2-Klein Transformer to ONNX.

Usage:
    venv\\Scripts\\python tools\\export_onnx.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
from safetensors.torch import load_file

MODEL_DIR = Path("models/base")
CONFIG_DIR = Path("models/base-config")
CHECKPOINT = MODEL_DIR / "flux-2-klein-4b.safetensors"
TRANSFORMER_CONFIG = CONFIG_DIR / "transformer" / "config.json"


# ---------------------------------------------------------------------------
# RMSNorm workaround for ONNX export (aten::rms_norm is not supported)
# ---------------------------------------------------------------------------
class ManualRMSNorm(nn.Module):
    """ONNX-compatible replacement for nn.RMSNorm."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x


def replace_rmsnorm(module: nn.Module):
    """Recursively replace nn.RMSNorm with ManualRMSNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.RMSNorm):
            manual = ManualRMSNorm(
                child.normalized_shape,
                child.eps,
                child.elementwise_affine,
            )
            if child.weight is not None:
                manual.weight.data.copy_(child.weight.data)
            setattr(module, name, manual)
        else:
            replace_rmsnorm(child)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_transformer(checkpoint: Path, config_path: Path):
    print(f"[export] Loading transformer config: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print(f"  num_layers (double): {config['num_layers']}")
    print(f"  num_single_layers: {config['num_single_layers']}")
    print(f"  num_attention_heads: {config['num_attention_heads']}")
    print(f"  attention_head_dim: {config['attention_head_dim']}")

    print(f"\n[export] Loading weights: {checkpoint}")
    state_dict = load_file(str(checkpoint), device="cpu")
    print(f"  Keys in checkpoint: {len(state_dict)}")

    print("[export] Converting checkpoint keys to diffusers format...")
    state_dict = convert_flux2_transformer_checkpoint_to_diffusers(state_dict)
    print(f"  Converted keys: {len(state_dict)}")

    print("\n[export] Building model...")
    model = Flux2Transformer2DModel.from_config(config)
    model.load_state_dict(state_dict, strict=True)
    print("  State dict loaded (strict=True) — all keys matched!")
    return model


def build_dummy_inputs(device, batch_size, height, width,
                       txt_seq_len, joint_attention_dim,
                       in_channels, guidance_embeds):
    """Construct dummy inputs matching the Flux2 pipeline."""
    img_seq_len = height * width

    hidden_states = torch.randn(
        batch_size, img_seq_len, in_channels,
        dtype=torch.float32, device=device,
    )
    encoder_hidden_states = torch.randn(
        batch_size, txt_seq_len, joint_attention_dim,
        dtype=torch.float32, device=device,
    )
    timestep = torch.tensor([0.5] * batch_size, dtype=torch.float32, device=device)

    # 4D coords matching cartesian_prod
    img_ids = torch.zeros(batch_size, img_seq_len, 4, dtype=torch.int64, device=device)
    idx = 0
    for t in range(batch_size):
        for h in range(height):
            for w in range(width):
                for layer in range(1):
                    img_ids[t, idx] = torch.tensor([t, h, w, layer], dtype=torch.int64)
                    idx += 1

    txt_ids = torch.zeros(batch_size, txt_seq_len, 4, dtype=torch.int64, device=device)
    idx = 0
    for t in range(batch_size):
        for h in range(1):
            for w in range(1):
                for ll in range(txt_seq_len):
                    txt_ids[t, idx] = torch.tensor([t, h, w, ll], dtype=torch.int64)
                    idx += 1

    # Mask for spatial cache: 0=skip, 1=execute, 2=execute+update
    # Shape: [batch_size, txt_seq_len + img_seq_len]
    full_seq_len = txt_seq_len + img_seq_len
    mask = torch.ones(batch_size, full_seq_len, dtype=torch.int32, device=device) * 2

    inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "mask": mask,
    }
    if guidance_embeds:
        inputs["guidance"] = torch.tensor([3.0] * batch_size, dtype=torch.float32, device=device)
    return inputs


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
            "mask": {0: "batch"},
        }
        print("\n[export] Fixed-shape export (seq_len not dynamic) — TensorRT compatible")
    else:
        dynamic_axes = {
            "hidden_states": {0: "batch", 1: "img_seq_len"},
            "encoder_hidden_states": {0: "batch", 1: "txt_seq_len"},
            "timestep": {0: "batch"},
            "img_ids": {0: "batch", 1: "img_seq_len"},
            "txt_ids": {0: "batch", 1: "txt_seq_len"},
            "mask": {0: "batch", 1: "seq_len"},
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
                onnx_program.save(str(output_path), save_as_external_data=True)
            print(f"[export] dynamo_export succeeded: {output_path}")
        except (torch.onnx.errors.OnnxExporterError, RuntimeError, ImportError) as e:
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
                        help="Disable dynamic axes for sequence lengths (needed for TensorRT)")
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