#!/usr/bin/env python3
# Copyright 2026 FluxRT-CPP Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Export per-block weights from a Flux2 safetensors checkpoint to raw binary files.
# Usage:
#   python tools/export_block_weights.py \
#       --checkpoint models/base/flux-2-klein-4b.safetensors \
#       --out-dir models/block_weights

import argparse
import os

import torch
from safetensors.torch import load_file


def save_tensor(t: torch.Tensor, path: str):
    """Save a float32 tensor as raw little-endian bytes."""
    t_f32 = t.detach().cpu().to(torch.float32).contiguous()
    with open(path, "wb") as f:
        f.write(t_f32.numpy().tobytes())


def export_double_block(state_dict, layer_idx, out_dir):
    """Export all weights for a double-stream transformer block."""
    prefix = f"transformer_blocks.{layer_idx}."
    img_prefix = prefix + "img_attn."
    txt_prefix = prefix + "txt_attn."
    img_mlp_prefix = prefix + "img_mlp."
    txt_mlp_prefix = prefix + "txt_mlp."

    # Collect all weight keys
    weights = {}
    for key in ["to_q", "to_k", "to_v", "to_out.0"]:
        weights[f"img_attn_{key}"] = state_dict[img_prefix + f"{key}.weight"]
        if img_prefix + f"{key}.bias" in state_dict:
            weights[f"img_attn_{key}_bias"] = state_dict[img_prefix + f"{key}.bias"]

    for key in ["to_q", "to_k", "to_v", "to_out.0"]:
        weights[f"txt_attn_{key}"] = state_dict[txt_prefix + f"{key}.weight"]
        if txt_prefix + f"{key}.bias" in state_dict:
            weights[f"txt_attn_{key}_bias"] = state_dict[txt_prefix + f"{key}.bias"]

    # MLP weights (net[0] and net[2] in FeedForward)
    for key in ["0", "2"]:
        weights[f"img_mlp_{key}"] = state_dict[img_mlp_prefix + f"{key}.weight"]
        if img_mlp_prefix + f"{key}.bias" in state_dict:
            weights[f"img_mlp_{key}_bias"] = state_dict[img_mlp_prefix + f"{key}.bias"]

    for key in ["0", "2"]:
        weights[f"txt_mlp_{key}"] = state_dict[txt_mlp_prefix + f"{key}.weight"]
        if txt_mlp_prefix + f"{key}.bias" in state_dict:
            weights[f"txt_mlp_{key}_bias"] = state_dict[txt_mlp_prefix + f"{key}.bias"]

    # QK norms (RMSNorm)
    for key in ["norm_q", "norm_k"]:
        weights[f"img_attn_{key}"] = state_dict[img_prefix + f"{key}.weight"]
        weights[f"txt_attn_{key}"] = state_dict[txt_prefix + f"{key}.weight"]

    # AdaGroupNorm / AdaLayerNorm modulation weights
    # These are typically named "norm.linear.weight" and "norm.linear.bias"
    for stream in ["img", "txt"]:
        mod_prefix = prefix + f"{stream}_norm.linear."
        if mod_prefix + "weight" in state_dict:
            weights[f"{stream}_norm_linear"] = state_dict[mod_prefix + "weight"]
            weights[f"{stream}_norm_linear_bias"] = state_dict[mod_prefix + "bias"]

    block_dir = os.path.join(out_dir, f"double_block_{layer_idx}")
    os.makedirs(block_dir, exist_ok=True)

    for name, tensor in weights.items():
        save_tensor(tensor, os.path.join(block_dir, f"{name}.bin"))

    print(f"  Exported {len(weights)} tensors to {block_dir}")
    return weights


def export_single_block(state_dict, layer_idx, out_dir):
    """Export all weights for a single-stream transformer block."""
    prefix = f"single_transformer_blocks.{layer_idx}."
    attn_prefix = prefix + "attn."

    weights = {}

    # Flux2ParallelSelfAttention weights
    weights["attn_to_qkv_mlp_proj"] = state_dict[attn_prefix + "to_qkv_mlp_proj.weight"]
    if attn_prefix + "to_qkv_mlp_proj.bias" in state_dict:
        weights["attn_to_qkv_mlp_proj_bias"] = state_dict[attn_prefix + "to_qkv_mlp_proj.bias"]

    weights["attn_to_out"] = state_dict[attn_prefix + "to_out.weight"]
    if attn_prefix + "to_out.bias" in state_dict:
        weights["attn_to_out_bias"] = state_dict[attn_prefix + "to_out.bias"]

    # RMSNorm weights
    for key in ["norm_q", "norm_k"]:
        weights[f"attn_{key}"] = state_dict[attn_prefix + f"{key}.weight"]

    # AdaGroupNorm modulation
    mod_prefix = prefix + "norm.linear."
    if mod_prefix + "weight" in state_dict:
        weights["norm_linear"] = state_dict[mod_prefix + "weight"]
        weights["norm_linear_bias"] = state_dict[mod_prefix + "bias"]

    block_dir = os.path.join(out_dir, f"single_block_{layer_idx}")
    os.makedirs(block_dir, exist_ok=True)

    for name, tensor in weights.items():
        save_tensor(tensor, os.path.join(block_dir, f"{name}.bin"))

    print(f"  Exported {len(weights)} tensors to {block_dir}")
    return weights


def export_final_layers(state_dict, out_dir):
    """Export final norm and projection layers."""
    weights = {}
    keys = [
        ("norm_out.linear.weight", "norm_out_linear"),
        ("norm_out.linear.bias", "norm_out_linear_bias"),
        ("proj_out.weight", "proj_out"),
        ("proj_out.bias", "proj_out_bias"),
    ]
    for src_name, dst_name in keys:
        if src_name in state_dict:
            weights[dst_name] = state_dict[src_name]

    block_dir = os.path.join(out_dir, "final_layers")
    os.makedirs(block_dir, exist_ok=True)

    for name, tensor in weights.items():
        save_tensor(tensor, os.path.join(block_dir, f"{name}.bin"))

    print(f"  Exported {len(weights)} tensors to {block_dir}")
    return weights


def main():
    parser = argparse.ArgumentParser(description="Export Flux2 block weights to binary")
    parser.add_argument("--checkpoint", required=True, help="Path to .safetensors checkpoint")
    parser.add_argument("--out-dir", default="models/block_weights", help="Output directory")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_file(args.checkpoint)
    print(f"Loaded {len(state_dict)} tensors")

    os.makedirs(args.out_dir, exist_ok=True)

    # Count blocks
    num_double = max(
        int(k.split("transformer_blocks.")[1].split(".")[0])
        for k in state_dict if k.startswith("transformer_blocks.")
    ) + 1

    num_single = max(
        int(k.split("single_transformer_blocks.")[1].split(".")[0])
        for k in state_dict if k.startswith("single_transformer_blocks.")
    ) + 1

    print(f"\nExporting {num_double} double-stream blocks...")
    for i in range(num_double):
        export_double_block(state_dict, i, args.out_dir)

    print(f"\nExporting {num_single} single-stream blocks...")
    for i in range(num_single):
        export_single_block(state_dict, i, args.out_dir)

    print(f"\nExporting final layers...")
    export_final_layers(state_dict, args.out_dir)

    # Save a metadata file with shapes
    meta_path = os.path.join(args.out_dir, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"num_double_blocks: {num_double}\n")
        f.write(f"num_single_blocks: {num_single}\n")
        for name, tensor in state_dict.items():
            if "transformer_blocks" in name or "single_transformer_blocks" in name:
                f.write(f"{name}: {list(tensor.shape)}\n")

    print(f"\nDone. Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
