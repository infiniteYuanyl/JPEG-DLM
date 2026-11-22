# cores/inference/generate.py

Source: `cores/inference/generate.py`.

This module owns the release CLI for offline generation: load the model
configuration, tokenizer and checkpoint, call the model sampler, decode native
token IDs, and write one `predictions_seed<N>.jsonl` per seed.

Checkpoint resolution is delegated to `utils/checkpoints.py`. Existing local
files are loaded directly. Missing files whose names match one of the three
released checkpoints are downloaded from `Seele-Yuan/JPEG-DLM` and checked
against the release SHA256 before model loading. Other missing paths raise
`FileNotFoundError`. Local checkpoints are never replaced by the resolver.

`generate_texts` owns batching, seed setup and output assembly. LM1B seeds once
per generation call and uses the random time grid; OWT reseeds each batch with
`seed + sample_offset` and uses the quantile grid. Each row contains the decoded
`prediction` and the original `token_ids`.

Changes to loading or CLI behavior belong here. Sampling mathematics and model
architecture belong in the existing model/module implementations. Validate
loading changes with `tests/test_checkpoint_loading.py`, CLI help, and a small
generation smoke check.
