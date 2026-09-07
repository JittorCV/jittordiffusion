#!/usr/bin/env python
# coding=utf-8

import sys
import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Default config based on:
# configs/config_moe.yaml
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config_moe.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="StyleExpert MoE inference (default config).")
    parser.add_argument("--content_path", type=str, required=True, help="Content image path.")
    parser.add_argument("--style_path", type=str, required=True, help="Style image path.")
    parser.add_argument("--output_path", type=str, default="outputs/styleexpert_out.png", help="Save path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Config yaml path.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt override.")
    parser.add_argument("--cpu", action="store_true", help="Force the Jittor CPU backend.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cpu:
        os.environ["STYLEEXPERT_FORCE_CPU"] = "1"

    from infer_core import inference

    image = inference(args.content_path, args.style_path, args.config, seed=args.seed, prompt=args.prompt)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
