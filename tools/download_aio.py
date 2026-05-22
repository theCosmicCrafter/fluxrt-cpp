"""
Download FLUX.2-klein-AIO from CivitAI.

This script is a build-time tool only (Constitution Article III).
It needs a CivitAI API key (read scope) to authenticate the download since
AIO lives on civitai.red (mature-content variant).

Usage:
    # Set CIVITAI_API_KEY in env, OR pass --api-key.
    export CIVITAI_API_KEY=...     # bash / zsh
    $env:CIVITAI_API_KEY = "..."   # PowerShell

    python tools/download_aio.py \\
        --model-version-id 2618128 \\
        --output models/aio/flux2-klein-aio.safetensors

Defaults to v1 of model 2327389 (the BF16 single-file). Pass a different
--model-version-id to grab another variant (e.g., the FP8 distill from
sibling model 2457796).

CivitAI download API:
    GET /api/v1/model-versions/{id}        -> JSON metadata incl. download URL
    GET https://civitai.com/api/download/models/{version_id}?token=XXX
        -> follows 302 to the actual file (CDN-signed)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


CIVITAI_API_BASE = "https://civitai.com/api/v1"
DEFAULT_MODEL_ID = 2327389
DEFAULT_VERSION_ID = 2618128  # AIO v1, BF16 single-file


def fetch_version_metadata(version_id: int, api_key: str) -> dict:
    """Fetch metadata for a CivitAI model version.

    Returns a dict including 'files' (list with download URLs) and
    'downloadUrl' (the canonical URL).
    """
    import requests  # noqa: PLC0415

    url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"[download_aio] Fetching metadata: {url}")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def stream_download(url: str, dest: Path, api_key: str) -> None:
    """Stream-download a (possibly large) file with progress."""
    import requests  # noqa: PLC0415

    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"[download_aio] GET {url}")
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk = 1024 * 1024  # 1 MB
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for piece in r.iter_content(chunk_size=chunk):
                if not piece:
                    continue
                f.write(piece)
                downloaded += len(piece)
                if total > 0:
                    pct = 100.0 * downloaded / total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(
                        f"\r[download_aio] {mb:8.1f} / {total_mb:8.1f} MB "
                        f"({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
    print(f"[download_aio] Wrote {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version-id", type=int, default=DEFAULT_VERSION_ID,
        help=f"CivitAI model-version ID (default: {DEFAULT_VERSION_ID}).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/aio/flux2-klein-aio.safetensors"),
        help="Destination .safetensors path.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="CivitAI API key. Defaults to $CIVITAI_API_KEY.",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Only fetch + print metadata; do not download.",
    )
    args = parser.parse_args()

    api_key: Optional[str] = args.api_key or os.environ.get("CIVITAI_API_KEY")
    if not api_key:
        print(
            "ERROR: CivitAI API key not provided. Pass --api-key or set "
            "the CIVITAI_API_KEY environment variable.",
            file=sys.stderr,
        )
        return 2

    meta = fetch_version_metadata(args.model_version_id, api_key)

    print(f"[download_aio] name:       {meta.get('name', '?')}")
    print(f"[download_aio] modelId:    {meta.get('modelId', '?')}")
    print(f"[download_aio] baseModel:  {meta.get('baseModel', '?')}")
    files = meta.get("files", [])
    for i, f in enumerate(files):
        size_mb = f.get("sizeKB", 0) / 1024
        print(
            f"[download_aio]   file[{i}]: name={f.get('name')} "
            f"format={f.get('metadata', {}).get('format')} "
            f"size={size_mb:.1f} MB"
        )

    if args.metadata_only:
        return 0

    if not files:
        print("ERROR: no files in version metadata.", file=sys.stderr)
        return 1

    # Use the first SafeTensor file by default.
    safetensors = [
        f for f in files
        if f.get("metadata", {}).get("format", "").lower() == "safetensor"
        or f.get("name", "").lower().endswith(".safetensors")
    ]
    primary = safetensors[0] if safetensors else files[0]
    download_url = primary.get("downloadUrl") or meta.get("downloadUrl")
    if not download_url:
        print("ERROR: no downloadUrl in metadata.", file=sys.stderr)
        return 1

    stream_download(download_url, args.output, api_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
