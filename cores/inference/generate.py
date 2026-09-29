from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

from cores.models.jpeg_dlm import JPEGDLM
from utils.data import decode_token_rows, load_tokenizer


def _autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def generate_texts(
    model,
    tokenizer,
    *,
    num_samples: int,
    steps: int,
    sc_scale: float,
    batch_size: int,
    seed: int,
    device: str,
    precision: str = "bf16",
    line_boundary_id: Optional[int] = None,
    sampling: str = "lm1b",
) -> list[dict[str, Any]]:
    """Generate token rows and decode them to text.

    ``lm1b`` seeds the random generators once and draws a new random time grid
    for every batch. ``owt1024`` reseeds them before every batch with ``seed``
    plus the index of the batch's first sample and uses the fixed quantile
    time grid.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    owt1024 = sampling == "owt1024"
    torch.manual_seed(seed)

    rows: list[dict[str, Any]] = []
    while len(rows) < num_samples:
        current = min(batch_size, num_samples - len(rows))
        if owt1024:
            torch.manual_seed(seed + len(rows))
        with _autocast_context(torch.device(device), precision):
            token_ids = model.sample(current, steps=steps, sc_scale=sc_scale, quantile_grid=owt1024)
        token_rows = token_ids.cpu().tolist()
        texts = decode_token_rows(tokenizer, token_rows, line_boundary_id)
        for text, ids in zip(texts, token_rows):
            rows.append({"prediction": text, "token_ids": ids})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JPEG-DLM samples.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file (.safetensors).")
    parser.add_argument("--config", required=True, help="Model configuration (configs/*.yaml).")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory; one predictions_seed<N>.jsonl file is written per seed.",
    )
    parser.add_argument(
        "--sampling",
        choices=["lm1b", "owt1024"],
        default="lm1b",
        help="Sampling procedure of the paper for the dataset (owt1024 with the OWT config).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sc-scale", type=float, default=3.0, help="Self-conditioning scale.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Also changes the samples; the paper uses 32 (LM1B) and 16 (OWT)."
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default="bf16",
        help="bf16 autocast on CUDA; other devices run in fp32.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    tokenizer = load_tokenizer(cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    model = JPEGDLM.from_pretrained(args.checkpoint, cfg, device=str(device))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rows = generate_texts(
            model,
            tokenizer,
            num_samples=args.num_samples,
            steps=args.steps,
            sc_scale=args.sc_scale,
            batch_size=args.batch_size,
            seed=seed,
            device=str(device),
            precision=args.precision,
            line_boundary_id=cfg.data.get("line_boundary_id", None),
            sampling=args.sampling,
        )
        path = output / f"predictions_seed{seed}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"seed={seed} samples={len(rows)} output={path}", flush=True)


if __name__ == "__main__":
    main()
