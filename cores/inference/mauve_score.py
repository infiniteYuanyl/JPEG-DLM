"""MAUVE of generated samples against the reference texts in references/.

Reference and generated texts are cut to their first 96 (LM1B) or 768
(OWT-1024) GPT-2 tokens and featurized with GPT-2 Large. Blank texts are
skipped.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

REFERENCES = Path(__file__).resolve().parents[2] / "references"
SCORER = "gpt2-large"
SEED = 42
# Reference file, GPT-2 tokens kept per text and featurization batch size.
SETTINGS = {
    "lm1b": ("lm1b_t5_mauve96.jsonl", 96, 16),
    "owt1024": ("owt1024_t5_mauve768.jsonl", 768, 1),
}


def read_texts(path: Path, field: str) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        texts = [str(json.loads(line)[field]) for line in handle if line.strip()]
    return [text for text in texts if text.strip()]


def device_id(device: str) -> int:
    """Map ``cpu``, ``cuda`` or ``cuda:N`` to the device id used by mauve-text."""

    dev = torch.device(device)
    if dev.type == "cpu":
        return -1
    index = 0 if dev.index is None else int(dev.index)
    # mauve-text silently falls back to the CPU for a device it cannot see.
    if dev.type != "cuda" or index >= torch.cuda.device_count():
        raise RuntimeError(f"{device} is not an available device")
    return index


def featurize(texts: list[str], max_len: int, device: int, batch_size: int, name: str):
    from mauve.compute_mauve import get_features_from_input

    return get_features_from_input(None, None, texts, SCORER, max_len, device, name, batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAUVE of generated samples with GPT-2 Large features.")
    parser.add_argument("--dataset", choices=list(SETTINGS), required=True)
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction files written by generate.")
    parser.add_argument("--output", required=True, help="Output directory for summary.json.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import mauve

    reference, max_len, batch_size = SETTINGS[args.dataset]
    device = device_id(args.device)
    p_features = featurize(read_texts(REFERENCES / reference, "reference"), max_len, device, batch_size, "p")

    per_file = []
    for pred in args.predictions:
        q_features = featurize(read_texts(Path(pred), "prediction"), max_len, device, batch_size, "q")
        result = mauve.compute_mauve(p_features=p_features, q_features=q_features, seed=SEED)
        per_file.append({"predictions": pred, "mauve": float(result.mauve)})
        print(f"{pred}: mauve={result.mauve:.4f}", flush=True)

    values = [row["mauve"] for row in per_file]
    summary = {
        "dataset": args.dataset,
        "per_file": per_file,
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"mean over {len(values)} file(s): mauve={summary['mean']:.4f} +/- {summary['std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
