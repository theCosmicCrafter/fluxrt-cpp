# Model Weights

Large model weight files belong here. They are **gitignored** (see `.gitignore`)
because they are too big for git and licensed separately.

## Required for Phase 0

```
models/
└── aio/
    └── flux2-klein-aio.safetensors    # ~16 GB BF16, from CivitAI
```

Get it via:

```powershell
$env:CIVITAI_API_KEY = "..."
python tools/download_aio.py
```

Or download manually from
https://civitai.com/models/2327389/flux2-klein-aio (redirects to civitai.red)
and place the file at `models/aio/flux2-klein-aio.safetensors`.

## Phase-by-phase requirements

| Phase | Required files |
|-------|---------------|
| 0 spike | `aio/flux2-klein-aio.safetensors` |
| 1 inference | + `aio/scheduler_config.json` (extracted) |
| 2 pipeline | (none new) |
| 3 streaming | (none new) |
| 4 GUI / LoRA | + `loras/*.safetensors` (any compatible LoRAs) |
| 5 quantization | + `calibration/*.npz` (calibration data, can regenerate) |

## Why CivitAI and not Hugging Face?

The AIO finetune is hosted on CivitAI by [MorikoMorizz](https://civitai.com/user/MorikoMorizz)
under Apache-2.0. There's no canonical HF mirror as of writing. The
`codeShare/FLUX.2-klein-AIO-SDNQ-4bit-dynamic` HF repo is a 4-bit quant
derivative, not the BF16 source.

## License

The AIO weights are licensed Apache-2.0 (per the CivitAI page). Local
modifications and redistribution are permitted but please retain
attribution to the upstream creator.
