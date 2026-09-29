from __future__ import annotations

from typing import Any, Optional, Sequence


def load_tokenizer(cfg: Any):
    """Load the slow T5 tokenizer at the pinned revision."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(cfg.data.tokenizer),
        revision=cfg.data.tokenizer_revision,
        use_fast=False,
        legacy=True,
    )


def decode_token_rows(
    tokenizer,
    rows: Sequence[Sequence[int]],
    line_boundary_id: Optional[int] = None,
) -> list[str]:
    """Decode generated token rows to text, dropping special tokens.

    With ``line_boundary_id`` (LM1B), each row is split at that token, every
    line is decoded with its whitespace collapsed, and non-empty lines are
    joined with single spaces. Without it (OWT), each row is decoded as one
    text and each whitespace run becomes one space.
    """

    if line_boundary_id is None:
        return [" ".join(tokenizer.decode(row, skip_special_tokens=True).split()) for row in rows]
    boundary = int(line_boundary_id)
    texts: list[str] = []
    for row in rows:
        segments: list[list[int]] = [[]]
        for token_id in row:
            if token_id == boundary:
                segments.append([])
            else:
                segments[-1].append(token_id)
        decoded = [tokenizer.decode(segment, skip_special_tokens=True) for segment in segments]
        pieces = [" ".join(text.split()) for text in decoded]
        texts.append(" ".join(piece for piece in pieces if piece))
    return texts
