#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent
STYLEEXPERT_REPO_ID = "HH-LG/StyleExpert"
STYLEEXPERT_REVISION = "main"
STYLEEXPERT_LOCAL_DIR = BASE_DIR
STYLEEXPERT_ALLOW_PATTERNS = ["weights/*"]

KONTEXT_REPO_ID = "black-forest-labs/FLUX.1-Kontext-dev"
KONTEXT_REVISION = "main"
KONTEXT_LOCAL_DIR = BASE_DIR / "models" / "FLUX.1-Kontext-dev"

SIGLIP_REPO_ID = "google/siglip-so400m-patch14-384"
SIGLIP_REVISION = "main"
SIGLIP_LOCAL_DIR = BASE_DIR / "models" / "siglip-so400m-patch14-384"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all pretrained model files required by StyleExpert inference."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, the local HF login state will be used.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--skip-styleexpert",
        action="store_true",
        help="Skip downloading StyleExpert adapter weights.",
    )
    parser.add_argument(
        "--skip-kontext",
        action="store_true",
        help="Skip downloading FLUX.1-Kontext-dev.",
    )
    parser.add_argument(
        "--skip-siglip",
        action="store_true",
        help="Skip downloading SigLIP.",
    )
    return parser


def download_snapshot(repo_id: str, revision: str, local_dir: Path, token: str | None, cache_dir: str | None, allow_patterns=None):
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"repo_id      : {repo_id}")
    print(f"revision     : {revision}")
    print(f"local_dir    : {local_dir}")
    print(f"allow_filter : {allow_patterns if allow_patterns is not None else 'ALL FILES'}")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
        local_dir=str(local_dir),
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )
    print(f"downloaded_to: {snapshot_path}")
    print("")


def main() -> None:
    args = build_parser().parse_args()

    if args.skip_styleexpert and args.skip_kontext and args.skip_siglip:
        raise ValueError("All download targets are skipped. Remove at least one --skip-* flag.")

    print("This script downloads the fixed model repos used by open_source/StyleExpert inference.")
    print("If FLUX.1-Kontext-dev is gated on Hugging Face, please make sure your account has accepted its license.")
    print("")

    if not args.skip_styleexpert:
        download_snapshot(
            repo_id=STYLEEXPERT_REPO_ID,
            revision=STYLEEXPERT_REVISION,
            local_dir=STYLEEXPERT_LOCAL_DIR,
            token=args.token,
            cache_dir=args.cache_dir,
            allow_patterns=STYLEEXPERT_ALLOW_PATTERNS,
        )

    if not args.skip_kontext:
        download_snapshot(
            repo_id=KONTEXT_REPO_ID,
            revision=KONTEXT_REVISION,
            local_dir=KONTEXT_LOCAL_DIR,
            token=args.token,
            cache_dir=args.cache_dir,
            allow_patterns=None,
        )

    if not args.skip_siglip:
        download_snapshot(
            repo_id=SIGLIP_REPO_ID,
            revision=SIGLIP_REVISION,
            local_dir=SIGLIP_LOCAL_DIR,
            token=args.token,
            cache_dir=args.cache_dir,
            allow_patterns=None,
        )


if __name__ == "__main__":
    main()
