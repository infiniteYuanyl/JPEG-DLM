#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Paper settings. Options given on the command line, such as --checkpoint,
# --steps or --sc-scale, override them.
PYTHONPATH=. python -m cores.inference.generate \
  --config configs/jpeg_dlm_lm1b_r0.25.yaml \
  --checkpoint checkpoints/jpeg_dlm_lm1b_r0.25.safetensors \
  --output outputs/lm1b_generations \
  --sampling lm1b \
  --seeds 0 1 2 3 4 \
  --num-samples 1024 \
  --steps 32 \
  --sc-scale 3.0 \
  --batch-size 32 \
  --precision bf16 \
  "$@"
