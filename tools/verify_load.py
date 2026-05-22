"""Phase 0.4: Verify base Klein-4B loads and inspect architecture.

Usage:
    venv/Scripts/python tools/verify_load.py
"""
import json
import sys
from pathlib import Path

import torch
from diffusers import Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
from safetensors.torch import load_file

MODEL_DIR = Path("models/base")
CONFIG_DIR = Path("models/base-config")
CHECKPOINT = MODEL_DIR / "flux-2-klein-4b.safetensors"
TRANSFORMER_CONFIG = CONFIG_DIR / "transformer" / "config.json"


def load_transformer():
    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT}")
        sys.exit(1)

    print(f"Loading transformer config: {TRANSFORMER_CONFIG}")
    with open(TRANSFORMER_CONFIG) as f:
        config = json.load(f)

    print(f"  num_layers (double): {config['num_layers']}")
    print(f"  num_single_layers: {config['num_single_layers']}")
    print(f"  num_attention_heads: {config['num_attention_heads']}")
    print(f"  attention_head_dim: {config['attention_head_dim']}")
    print(f"  mlp_ratio: {config['mlp_ratio']}")
    print(f"  patch_size: {config['patch_size']}")
    print(f"  in_channels: {config['in_channels']}")
    print(f"  guidance_embeds: {config['guidance_embeds']}")

    print(f"\nLoading weights from: {CHECKPOINT}")
    state_dict = load_file(str(CHECKPOINT), device="cpu")
    print(f"  Keys in checkpoint: {len(state_dict)}")

    print("\nConverting checkpoint keys to diffusers format...")
    state_dict = convert_flux2_transformer_checkpoint_to_diffusers(state_dict)
    print(f"  Converted keys: {len(state_dict)}")

    print("\nInstantiating Flux2Transformer2DModel...")
    model = Flux2Transformer2DModel.from_config(config)
    model.load_state_dict(state_dict, strict=True)
    print("  State dict loaded (strict=True) — all keys matched!")

    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def dummy_forward(model, device):
    print(f"\n--- Dummy forward pass on {device} ---")
    try:
        model.to(device)
    except RuntimeError as e:
        if "not compatible" in str(e).lower() or "sm_" in str(e):
            print(f"  CUDA incompatible ({e}). Falling back to CPU.")
            device = "cpu"
            model.to(device)
        else:
            raise
    model.eval()

    batch_size = 1
    height, width = 64, 64
    img_seq_len = height * width  # 4096 for 64x64

    # hidden_states is PACKED: [B, H*W, in_channels]
    # Generate latent in [B, C, H, W] then pack to [B, H*W, C]
    latents_4d = torch.randn(
        batch_size, model.config.in_channels, height, width,
        dtype=torch.float32, device=device,
    )
    hidden_states = latents_4d.reshape(
        batch_size, model.config.in_channels, img_seq_len,
    ).permute(0, 2, 1)

    # img_ids: [B, H*W, 4] — 4D coords (T, H, W, L)
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    layer_id = torch.arange(1)
    img_ids = torch.cartesian_prod(t, h, w, layer_id)
    img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    # Text embeddings: [B, seq_len, joint_attention_dim]
    txt_seq_len = 256
    encoder_hidden_states = torch.randn(
        batch_size, txt_seq_len, model.config.joint_attention_dim,
        dtype=torch.float32, device=device,
    )

    # txt_ids: [B, seq_len, 4]
    t = torch.arange(1)
    h = torch.arange(1)
    w = torch.arange(1)
    ll = torch.arange(txt_seq_len)
    txt_ids = torch.cartesian_prod(t, h, w, ll).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    # Timestep: scalar per sample (diffusers divides by 1000 internally)
    timestep = torch.tensor([0.5], dtype=torch.float32, device=device)

    if model.config.guidance_embeds:
        guidance = torch.tensor([3.0], dtype=torch.float32, device=device)
    else:
        guidance = None

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )

    if isinstance(output, tuple):
        output = output[0]
    print(f"  hidden_states:  {hidden_states.shape}")
    print(f"  img_ids:        {img_ids.shape}")
    print(f"  txt_ids:        {txt_ids.shape}")
    print(f"  Output shape:   {output.shape}")
    print("  OK Forward pass succeeded")
    return output


def main():
    model = load_transformer()
    total, trainable = count_params(model)
    print(f"\nParameters: {total:,} total ({total/1e6:.1f}M), {trainable:,} trainable")

    if torch.cuda.is_available():
        try:
            _ = torch.randn(1, device="cuda")
            device = "cuda"
        except RuntimeError:
            print("  CUDA probe failed (sm_120 incompatibility?). Using CPU.")
            device = "cpu"
    else:
        device = "cpu"
    dummy_forward(model, device)

    print("\nOK Phase 0.4 verification complete.")


if __name__ == "__main__":
    main()
