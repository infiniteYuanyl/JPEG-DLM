from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

SCORER = "gpt2-large"
MAX_LENGTH = 1024


def read_predictions(path: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [str(json.loads(line)["prediction"]) for line in handle if line.strip()]


def _entropy(token_ids: Sequence[int]) -> float:
    counts = Counter(token_ids)
    total = float(len(token_ids))
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


@torch.no_grad()
def score_gen_ppl(
    texts: Sequence[str],
    tokenizer,
    model,
    *,
    device: torch.device,
    batch_size: int = 8,
    prepend_bos: bool = False,
) -> dict[str, Any]:
    """Token-weighted Gen-PPL and mean per-text unigram entropy.

    Each text is scored on its first ``MAX_LENGTH`` tokens, and its first token
    is only context. With ``prepend_bos`` the scorer's BOS token
    (``<|endoftext|>`` for GPT-2) is put in front of the text as context, so
    its first ``MAX_LENGTH - 1`` tokens are all scored. Entropy is computed on
    the text tokens, without the BOS. Blank texts are skipped.
    """

    budget = MAX_LENGTH - 1 if prepend_bos else MAX_LENGTH
    encoded = [
        tokenizer.encode(text, add_special_tokens=False)[:budget] if text.strip() else []
        for text in texts
    ]
    entropies = [_entropy(row) for row in encoded if row]
    if prepend_bos:
        scorable = [[tokenizer.bos_token_id] + row for row in encoded if row]
    else:
        scorable = [row for row in encoded if len(row) >= 2]
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(scorable), batch_size):
        chunk = scorable[start : start + batch_size]
        max_len = max(len(row) for row in chunk)
        # GPT-2 has no padding token; padded positions are masked out.
        input_ids = torch.full((len(chunk), max_len), tokenizer.eos_token_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for row_i, row in enumerate(chunk):
            input_ids[row_i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention[row_i, : len(row)] = 1
        input_ids = input_ids.to(device)
        attention = attention.to(device)
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            logits = model(input_ids=input_ids, attention_mask=attention).logits
        shift_logits = logits.float()[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        mask = attention[:, 1:].to(torch.float32)
        losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape)
        row_nll = (losses * mask).sum(dim=1)
        row_tokens = mask.sum(dim=1)
        for row_i in range(len(chunk)):
            total_nll += float(row_nll[row_i].item())
            total_tokens += int(row_tokens[row_i].item())

    return {
        "gen_ppl": float(math.exp(total_nll / total_tokens)) if total_tokens else float("nan"),
        "entropy": float(statistics.fmean(entropies)) if entropies else float("nan"),
        "total_tokens": total_tokens,
        "num_skipped": len(texts) - len(scorable),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Gen-PPL and unigram entropy of generated samples with GPT-2 Large."
    )
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction files written by generate.")
    parser.add_argument("--output", required=True, help="Output directory for summary.json.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--prepend-bos",
        action="store_true",
        help="Put <|endoftext|> in front of each text as context, so that its first token is scored too.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(SCORER)
    model = AutoModelForCausalLM.from_pretrained(SCORER).to(device).eval()

    per_file = []
    for pred in args.predictions:
        row = {"predictions": pred}
        row.update(
            score_gen_ppl(
                read_predictions(pred),
                tokenizer,
                model,
                device=device,
                batch_size=args.batch_size,
                prepend_bos=args.prepend_bos,
            )
        )
        per_file.append(row)
        print(f"{pred}: gen_ppl={row['gen_ppl']:.4f} entropy={row['entropy']:.4f}", flush=True)

    summary: dict[str, Any] = {"per_file": per_file, "mean": {}, "std": {}}
    for key in ("gen_ppl", "entropy"):
        values = [row[key] for row in per_file]
        summary["mean"][key] = float(statistics.fmean(values))
        summary["std"][key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"mean over {len(per_file)} file(s): "
        f"gen_ppl={summary['mean']['gen_ppl']:.4f} +/- {summary['std']['gen_ppl']:.4f} "
        f"entropy={summary['mean']['entropy']:.4f} +/- {summary['std']['entropy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
