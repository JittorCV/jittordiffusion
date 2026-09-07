#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent
DATASET_REPO_ID = "HH-LG/StyleExpert"
DATASET_REVISION = "main"
DATASET_LOCAL_DIR = BASE_DIR / "datasets" / "StyleExpert"
METADATA_ONLY_PATTERNS = [
    "*.csv",
    "*.json",
    "*.jsonl",
    "*.md",
    "*.txt",
    "*.yaml",
    "*.yml",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the StyleExpert dataset from Hugging Face."
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
        "--metadata-only",
        action="store_true",
        help="Only download lightweight metadata files instead of the full dataset shards.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    local_dir = DATASET_LOCAL_DIR.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = METADATA_ONLY_PATTERNS if args.metadata_only else None

    print(f"repo_id      : {DATASET_REPO_ID}")
    print(f"revision     : {DATASET_REVISION}")
    print(f"local_dir    : {local_dir}")
    print(f"allow_filter : {allow_patterns if allow_patterns is not None else 'ALL FILES'}")

    snapshot_path = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        token=args.token,
        local_dir=str(local_dir),
        cache_dir=args.cache_dir,
        allow_patterns=allow_patterns,
    )

    print(f"downloaded_to: {snapshot_path}")


if __name__ == "__main__":
    main()
