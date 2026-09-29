from __future__ import annotations

from numbers import Integral
from typing import Any, Optional, Sequence

T5_TOKENIZER_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"


def _token_row(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    row = list(value)
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, Integral)
        for token_id in row
    ):
        raise TypeError("token rows must contain integer ids")
    return [int(token_id) for token_id in row]


def load_tokenizer(cfg: Any):
    """Load the pinned slow T5 tokenizer and assert the JPEG-DLM role ids."""

    from transformers import AutoTokenizer

    name = str(cfg.data.tokenizer)
    revision = cfg.data.get("tokenizer_revision", T5_TOKENIZER_REVISION)
    local_files_only = bool(cfg.data.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        use_fast=False,
        local_files_only=local_files_only,
    )
    expected = {
        "pad_token_id": 0,
        "eos_token_id": int(cfg.data.eos_id),
    }
    for attr, value in expected.items():
        actual = getattr(tokenizer, attr, None)
        if actual is None or int(actual) != int(value):
            raise RuntimeError(f"tokenizer {attr} expected {value}, got {actual}")
    # LM1B packing uses two T5 sentinels as role tokens; OWT packing uses none.
    for token, key in (("<extra_id_0>", "block_bos_id"), ("<extra_id_1>", "line_boundary_id")):
        expected_id = cfg.data.get(key, None)
        if expected_id is not None and int(tokenizer.convert_tokens_to_ids(token)) != int(expected_id):
            raise RuntimeError(f"T5 {token} id does not match {key}")
    return tokenizer


def decode_token_rows(
    tokenizer,
    rows: Sequence[Sequence[int]],
    line_boundary_id: Optional[int] = None,
    collapse_whitespace: bool = False,
) -> list[str]:
    """Decode packed token rows.

    LM1B rows are split on the line-boundary token, each line is decoded on its
    own, and the lines are joined with single spaces. Without a line-boundary
    token (OWT) each row is decoded as one text; EOS document separators are
    dropped as special tokens, and ``collapse_whitespace`` turns every run of
    whitespace into one space.
    """

    if line_boundary_id is None:
        texts = [tokenizer.decode(_token_row(value), skip_special_tokens=True) for value in rows]
        if collapse_whitespace:
            texts = [" ".join(text.split()) for text in texts]
        return texts
    boundary = int(line_boundary_id)
    texts: list[str] = []
    for value in rows:
        row = _token_row(value)
        segments: list[list[int]] = [[]]
        for token_id in row:
            if token_id == boundary:
                segments.append([])
            else:
                segments[-1].append(token_id)
        decoded = [
            tokenizer.decode(segment, skip_special_tokens=True)
            for segment in segments
        ]
        pieces = [" ".join(text.split()) for text in decoded]
        texts.append(" ".join(piece for piece in pieces if piece))
    return texts
