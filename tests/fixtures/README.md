# Parity Test Fixtures

Captured Python FluxRT outputs used as ground truth for C++ parity tests.

## Format

Each fixture is named `fixture_<id>_<seed>.npz` and contains:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `input_image` | `(H, W, 3)` | uint8 | Input frame (BGR) |
| `prompt` | `()` | UTF-8 string | Text prompt |
| `seed` | `()` | int64 | RNG seed |
| `steps` | `()` | int64 | Number of inference steps |
| `latent_step_0` | `(1, 16, h, w)` | float16 | Latent after step 0 |
| `latent_step_N` | `(1, 16, h, w)` | float16 | Latent after step N |
| `output_image` | `(H, W, 3)` | uint8 | Final decoded image |

`H, W` = configured resolution (default 320×576 or 512×512).
`h, w` = `H/8, W/8` (VAE downsamples 8×).

## Generation

Run `tools/capture_reference.py` against the upstream Python FluxRT to
populate this directory. See Phase 0 task 0.4.

## Why Captured (not Live-Compared)?

Capturing once and comparing offline:

1. Removes Python from the C++ test path (Article III).
2. Pins reference outputs across upstream FluxRT changes.
3. Allows running parity tests in isolated environments / CI.

## Gitignore

`.npz`, `.bin`, `.npy`, `.pt` are gitignored — too large for git. Either:

- Regenerate with `tools/capture_reference.py`, or
- Pull from a release artifact (added in Phase 1 if needed)
