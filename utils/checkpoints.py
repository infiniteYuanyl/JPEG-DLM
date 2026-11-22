"""Locate local JPEG-DLM weights or download a released checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "Seele-Yuan/JPEG-DLM"
CHECKPOINTS = {
    "lm1b-4x": (
        "jpeg_dlm_lm1b_r0.25.safetensors",
        "ac7ee3d10b2a55507869101221423f9245229eceb6cc0fb41c657724429fb76e",
    ),
    "lm1b-2x": (
        "jpeg_dlm_lm1b_r0.5.safetensors",
        "838a726c750ade08b932e62b4bf567d5cc0b3dc128214486fce8def7d1001fc8",
    ),
    "owt-2x": (
        "jpeg_dlm_owt1024_r0.5.safetensors",
        "c7f433d50ff0fab8ba096722d33ba98c8e0f8300853120a93c4541f062488bb5",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model: str, output_dir: Path, repo_id: str, revision: str) -> Path:
    filename, expected = CHECKPOINTS[model]
    destination = output_dir / filename
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or sha256(destination) != expected:
            raise RuntimeError(
                f"Existing checkpoint differs from the release: {destination}. "
                "Choose another --output-dir to keep this file."
            )
        return destination
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision,
            local_dir=str(output_dir),
        )
    )
    if sha256(downloaded) != expected:
        raise RuntimeError(f"Checkpoint checksum mismatch: {downloaded}")
    return downloaded


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.is_file():
        return path
    if path.exists() or path.is_symlink():
        raise FileNotFoundError(f"Checkpoint path is not a file: {path}")
    model = next(
        (name for name, (filename, _) in CHECKPOINTS.items() if filename == path.name),
        None,
    )
    if model is None:
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return download(model, path.parent, REPO_ID, "main")
