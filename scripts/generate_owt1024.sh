#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Paper settings. Options given on the command line, such as --checkpoint,
# --steps or --sc-scale, override them.
PYTHONPATH=. python -m cores.inference.generate \
  --config configs/jpeg_dlm_owt1024_r0.5.yaml \
  --checkpoint checkpoints/jpeg_dlm_owt1024_r0.5.safetensors \
  --output outputs/owt1024_generations \
  --sampling owt1024 \
  --seeds 0 1 2 3 4 \
  --num-samples 1024 \
  --steps 32 \
  --sc-scale 2.0 \
  --batch-size 16 \
  --precision bf16 \
  "$@"
