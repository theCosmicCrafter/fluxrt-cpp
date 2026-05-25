"""Phase 0.4: Capture deterministic PyTorch reference outputs for C++ parity testing.

Usage:
    venv\\Scripts\\python tools\\capture_reference.py

Saves binary tensors to tests/fixtures/ for PSNR comparison.
"""
import argparse
import json
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
OUT_DIR = Path("tests/fixtures")


def load_transformer():
    print(f"Loading transformer config: {TRANSFORMER_CONFIG}")
    with open(TRANSFORMER_CONFIG, encoding="utf-8") as f:
        config = json.load(f)

    print(f"Loading weights from: {CHECKPOINT}")
    state_dict = load_file(str(CHECKPOINT), device="cpu")
    print(f"  Keys in checkpoint: {len(state_dict)}")

    print("Converting checkpoint keys to diffusers format...")
    state_dict = convert_flux2_transformer_checkpoint_to_diffusers(state_dict)
    print(f"  Converted keys: {len(state_dict)}")

    print("Instantiating Flux2Transformer2DModel...")
    model = Flux2Transformer2DModel.from_config(config)
    model.load_state_dict(state_dict, strict=True)
    print("  State dict loaded (strict=True) — all keys matched!")
    return model


def save_raw(path: Path, tensor: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save in native dtype — C++ loader uses matching type (float32 for float, int64 for int64)
    arr = tensor.detach().cpu().numpy()
    arr.tofile(path)
    print(f"  {path.name}: shape={list(tensor.shape)}, dtype={tensor.dtype} (saved as {arr.dtype})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"[capture] Loading model on {device}...")
    model = load_transformer()
    model.to(device).eval()

    # Monkey-patch RMSNorm same as export for parity
    print("[capture] Replacing nn.RMSNorm with ManualRMSNorm...")

    class ManualRMSNorm(nn.Module):
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

    def _replace_rmsnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.RMSNorm):
                manual = ManualRMSNorm(child.normalized_shape, child.eps, child.elementwise_affine)
                if child.weight is not None:
                    manual.weight.data.copy_(child.weight.data)
                setattr(module, name, manual)
            else:
                _replace_rmsnorm(child)

    _replace_rmsnorm(model)
    print("  Done.")

    B = 1
    IMG_L = 4096
    TXT_L = 256

    print(f"[capture] Generating deterministic inputs (seed={args.seed})...")
    torch.manual_seed(args.seed)

    hidden_states = torch.randn(B, IMG_L, 128, dtype=torch.float32, device=device)
    encoder_hidden_states = torch.randn(B, TXT_L, 7680, dtype=torch.float32, device=device)
    timestep = torch.tensor([0.5], dtype=torch.float32, device=device)

    # img_ids matching cartesian_prod pattern from verify_load.py
    img_ids = torch.zeros(B, IMG_L, 4, dtype=torch.int64, device=device)
    idx = 0
    for t in range(1):
        for h in range(64):
            for w in range(64):
                for layer in range(1):
                    img_ids[0, idx] = torch.tensor([t, h, w, layer], dtype=torch.int64)
                    idx += 1

    # txt_ids matching cartesian_prod pattern
    txt_ids = torch.zeros(B, TXT_L, 4, dtype=torch.int64, device=device)
    idx = 0
    for t in range(1):
        for h in range(1):
            for w in range(1):
                for ll in range(TXT_L):
                    txt_ids[0, idx] = torch.tensor([t, h, w, ll], dtype=torch.int64)
                    idx += 1

    guidance = torch.tensor([3.0], dtype=torch.float32, device=device)

    print("[capture] Running forward pass...")
    kwargs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "return_dict": False,
    }
    if model.config.guidance_embeds:
        kwargs["guidance"] = guidance

    with torch.no_grad():
        output = model(**kwargs)[0]

    # Generate mask fixtures for spatial cache testing
    full_seq_len = TXT_L + IMG_L
    mask_all_active = torch.ones(B, full_seq_len, dtype=torch.int32, device="cpu") * 2
    mask_selective = mask_all_active.clone()
    # Skip first 1024 image tokens
    mask_selective[0, TXT_L:TXT_L + 1024] = 0

    print("[capture] Saving tensors...")
    save_raw(args.out_dir / "hidden_states.bin", hidden_states)
    save_raw(args.out_dir / "encoder_hidden_states.bin", encoder_hidden_states)
    save_raw(args.out_dir / "timestep.bin", timestep)
    save_raw(args.out_dir / "img_ids.bin", img_ids)
    save_raw(args.out_dir / "txt_ids.bin", txt_ids)
    save_raw(args.out_dir / "guidance.bin", guidance)
    save_raw(args.out_dir / "sample.bin", output)
    save_raw(args.out_dir / "mask_all_active.bin", mask_all_active)
    save_raw(args.out_dir / "mask_selective.bin", mask_selective)

    print(f"\n[capture] OK Phase 0.4 complete. Tensors saved to {args.out_dir}")


if __name__ == "__main__":
    main()
