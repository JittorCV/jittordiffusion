#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <content_path> <style_path> [output_path] [seed]"
  exit 1
fi

CONTENT_PATH="$1"
STYLE_PATH="$2"
OUTPUT_PATH="${3:-outputs/styleexpert_out.png}"
SEED="${4:-42}"

python "$ROOT_DIR/infer.py" \
  --content_path "$CONTENT_PATH" \
  --style_path "$STYLE_PATH" \
  --output_path "$OUTPUT_PATH" \
  --seed "$SEED"
