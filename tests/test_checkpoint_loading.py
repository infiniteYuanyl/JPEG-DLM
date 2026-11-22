from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import checkpoints


FIXTURE = b"test checkpoint payload"
FIXTURE_SHA = hashlib.sha256(FIXTURE).hexdigest()


def fake_download(**kwargs):
    path = Path(kwargs["local_dir"]) / kwargs["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FIXTURE)
    return str(path)


class CheckpointLoadingTests(unittest.TestCase):
    def test_local_checkpoint_is_used_without_network_or_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / checkpoints.CHECKPOINTS["lm1b-4x"][0]
            path.write_bytes(b"local custom weights")
            with patch.object(checkpoints, "hf_hub_download") as remote:
                self.assertEqual(checkpoints.resolve_checkpoint(path), path)
                remote.assert_not_called()
            self.assertEqual(path.read_bytes(), b"local custom weights")

    def test_each_released_variant_downloads_once_to_the_requested_directory(self):
        for model, (filename, _) in checkpoints.CHECKPOINTS.copy().items():
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "new weights" / filename
                with patch.dict(checkpoints.CHECKPOINTS, {model: (filename, FIXTURE_SHA)}, clear=True):
                    with patch.object(checkpoints, "hf_hub_download", side_effect=fake_download) as remote:
                        self.assertEqual(checkpoints.resolve_checkpoint(path), path)
                        self.assertEqual(checkpoints.resolve_checkpoint(path), path)
                        remote.assert_called_once_with(
                            repo_id=checkpoints.REPO_ID, filename=filename,
                            revision="main", local_dir=str(path.parent),
                        )
                self.assertEqual(path.read_bytes(), FIXTURE)

    def test_missing_custom_path_does_not_trigger_a_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-release.safetensors"
            with patch.object(checkpoints, "hf_hub_download") as remote:
                with self.assertRaises(FileNotFoundError):
                    checkpoints.resolve_checkpoint(path)
                remote.assert_not_called()
            self.assertFalse(path.exists())

    def test_directory_and_broken_symlink_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / checkpoints.CHECKPOINTS["lm1b-4x"][0]
            path.mkdir()
            with patch.object(checkpoints, "hf_hub_download") as remote:
                with self.assertRaises(FileNotFoundError):
                    checkpoints.resolve_checkpoint(path)
                self.assertTrue(path.is_dir())
                path.rmdir()
                path.symlink_to(Path(directory) / "missing-source")
                with self.assertRaises(FileNotFoundError):
                    checkpoints.resolve_checkpoint(path)
                self.assertTrue(path.is_symlink())
                remote.assert_not_called()

    def test_downloaded_checksum_is_checked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / checkpoints.CHECKPOINTS["lm1b-4x"][0]
            with patch.object(checkpoints, "hf_hub_download", side_effect=fake_download):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    checkpoints.resolve_checkpoint(path)
            self.assertEqual(path.read_bytes(), FIXTURE)

    def test_explicit_download_keeps_an_existing_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / checkpoints.CHECKPOINTS["lm1b-4x"][0]
            path.write_bytes(b"retain original file")
            with patch.object(checkpoints, "hf_hub_download") as remote:
                with self.assertRaises(RuntimeError):
                    checkpoints.download("lm1b-4x", path.parent, checkpoints.REPO_ID, "main")
                remote.assert_not_called()
            self.assertEqual(path.read_bytes(), b"retain original file")

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ["torch", "omegaconf", "safetensors", "transformers"]),
        "Generation smoke check needs the inference dependencies",
    )
    def test_generation_loads_the_resolved_checkpoint_and_writes_native_ids(self):
        import torch
        from cores.inference import generate

        class FixtureModel:
            def sample(self, count, **kwargs):
                return torch.zeros((count, 128), dtype=torch.long)

        filename = checkpoints.CHECKPOINTS["lm1b-4x"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights" / filename
            output = Path(directory) / "predictions"
            cfg = SimpleNamespace(data={"line_boundary_id": 32098})
            args = argparse.Namespace(
                checkpoint=str(path), config="fixture.yaml", output=str(output),
                device="cpu", seeds=[0], num_samples=1, steps=1, sc_scale=1.0,
                batch_size=1, precision="fp32", sampling="lm1b",
            )
            with patch.dict(checkpoints.CHECKPOINTS, {"lm1b-4x": (filename, FIXTURE_SHA)}, clear=True), \
                 patch.object(checkpoints, "hf_hub_download", side_effect=fake_download), \
                 patch.object(generate, "parse_args", return_value=args), \
                 patch.object(generate.OmegaConf, "load", return_value=cfg), \
                 patch.object(generate, "load_tokenizer", return_value=object()), \
                 patch.object(generate, "decode_token_rows", return_value=["test fixture"]), \
                 patch.object(generate.JPEGDLM, "from_pretrained", return_value=FixtureModel()) as load:
                generate.main()
            load.assert_called_once_with(path, cfg, device="cpu")
            rows = [json.loads(line) for line in (output / "predictions_seed0.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [{"prediction": "test fixture", "token_ids": [0] * 128}])


if __name__ == "__main__":
    unittest.main()
