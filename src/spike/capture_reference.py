"""Phase 0.4: Capture PyTorch reference I/O for parity testing."""
import argparse
from pathlib import Path
import torch
import os

from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel

def save_raw(path: Path, tensor: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Handle bf16 by converting to fp32 first, keep int64 as is
    if tensor.dtype in (torch.bfloat16, torch.float16):
        tensor = tensor.float()
    tensor.detach().cpu().numpy().tofile(path)
    print(f"Saved {path} (shape: {list(tensor.shape)}, dtype: {tensor.dtype})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../FluxRT/FLUX.2-klein-4B/transformer")
    parser.add_argument("--out-dir", type=Path, default="tests/fixtures")
    args = parser.parse_args()

    print(f"[capture] Loading {args.model}...")
    model = Flux2Transformer2DModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()
    model.to("cuda")

    B = 1
    IMG_L = 4096
    TXT_L = 256

    # Generate deterministic inputs matching Python's native pipeline
    torch.manual_seed(42)
    hidden_states = torch.randn(B, IMG_L, 128, dtype=torch.bfloat16, device="cuda")
    encoder_hidden_states = torch.randn(B, TXT_L, 7680, dtype=torch.bfloat16, device="cuda")
    timestep = torch.tensor([0.5], dtype=torch.bfloat16, device="cuda")
    
    # coords matching Python cartesian_prod (which outputs int64)
    img_ids = torch.zeros(B, IMG_L, 4, dtype=torch.int64, device="cuda")
    idx = 0
    for t in range(1):
        for h in range(64):
            for w in range(64):
                for l in range(1):
                    img_ids[0, idx] = torch.tensor([t, h, w, l], dtype=torch.int64)
                    idx += 1

    txt_ids = torch.zeros(B, TXT_L, 4, dtype=torch.int64, device="cuda")
    idx = 0
    for t in range(1):
        for h in range(1):
            for w in range(1):
                for l in range(256):
                    txt_ids[0, idx] = torch.tensor([t, h, w, l], dtype=torch.int64)
                    idx += 1

    guidance = torch.tensor([3.0], dtype=torch.bfloat16, device="cuda")

    print("[capture] Running forward pass...")
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False
        )[0]

    print("[capture] Saving tensors...")
    save_raw(args.out_dir / "hidden_states.bin", hidden_states)
    save_raw(args.out_dir / "encoder_hidden_states.bin", encoder_hidden_states)
    save_raw(args.out_dir / "timestep.bin", timestep)
    save_raw(args.out_dir / "img_ids.bin", img_ids)
    save_raw(args.out_dir / "txt_ids.bin", txt_ids)
    save_raw(args.out_dir / "guidance.bin", guidance)
    save_raw(args.out_dir / "sample.bin", output)

    print(f"[capture] OK Phase 0.4 complete. Tensors saved to {args.out_dir}")

if __name__ == "__main__":
    main()