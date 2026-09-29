#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/lm1b_eval}"
DEVICE="${DEVICE:-cuda}"
# faiss, used by MAUVE, can stall when it starts one thread per core.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# --output and --device apply to sampling and scoring; all other options, such
# as --checkpoint, are passed on to scripts/generate_lm1b.sh.
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --output=*) OUTPUT_DIR="${1#*=}"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

# Sample seeds 0 to 4 with the paper settings.
rm -f "$OUTPUT_DIR"/generations/predictions_seed*.jsonl
bash scripts/generate_lm1b.sh "${ARGS[@]}" --output "$OUTPUT_DIR/generations" --device "$DEVICE"

shopt -s nullglob
PREDICTIONS=("$OUTPUT_DIR"/generations/predictions_seed*.jsonl)
if [ "${#PREDICTIONS[@]}" -eq 0 ]; then
  echo "no prediction files found under $OUTPUT_DIR/generations" >&2
  exit 1
fi

PYTHONPATH=. python -m cores.inference.gen_ppl \
  --predictions "${PREDICTIONS[@]}" \
  --output "$OUTPUT_DIR/gen_ppl" \
  --device "$DEVICE" \
  --batch-size 8

PYTHONPATH=. python -m cores.inference.mauve_score \
  --dataset lm1b \
  --predictions "${PREDICTIONS[@]}" \
  --output "$OUTPUT_DIR/mauve" \
  --device "$DEVICE"
