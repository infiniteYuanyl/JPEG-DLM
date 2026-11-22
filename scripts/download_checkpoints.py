"""Download JPEG-DLM inference weights from Hugging Face."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.checkpoints import CHECKPOINTS, REPO_ID, download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*CHECKPOINTS, "all"], default="lm1b-4x")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    models = CHECKPOINTS if args.model == "all" else [args.model]
    for model in models:
        path = download(model, args.output_dir, args.repo_id, args.revision)
        print(f"Verified checkpoint: {path}")


if __name__ == "__main__":
    main()
