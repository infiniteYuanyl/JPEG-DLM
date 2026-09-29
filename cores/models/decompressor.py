from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .layers import (
    AbsolutePositionalEmbedding,
    DecoderBlock,
    FeatureBottleneckProjection,
    FeedForward,
    ScaleGate,
)


class Decompressor(nn.Module):
    """Restore token-length features from the compressed latent sequence."""

    def __init__(
        self,
        latent_dim: int = 512,
        feature_dim: int = 512,
        bottleneck_dim: int = 128,
        num_latents: int = 32,
        layers: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_len: int = 128,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_latents = int(num_latents)
        self.max_len = int(max_len)
        if self.latent_dim % int(heads) != 0:
            raise ValueError("latent_dim must be divisible by heads")
        dim_head = self.latent_dim // int(heads)

        self.pos_emb = AbsolutePositionalEmbedding(self.latent_dim, self.max_len)
        self.pos_latent = AbsolutePositionalEmbedding(self.latent_dim, self.num_latents)
        self.base_query = nn.Parameter(torch.empty(1, 1, self.latent_dim))
        nn.init.normal_(self.base_query, mean=0.0, std=0.02)
        self.query_ffn = FeedForward(self.latent_dim, mult=mlp_ratio, dropout=dropout)
        self.query_gate = ScaleGate()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim=self.latent_dim,
                    latent_dim=self.latent_dim,
                    hidden=self.latent_dim,
                    heads=int(heads),
                    dim_head=dim_head,
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(layers))
            ]
        )
        self.out_proj = FeatureBottleneckProjection(
            self.latent_dim,
            self.bottleneck_dim,
            self.feature_dim,
            bias=False,
        )

    def forward(self, latents: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(f"latents must be [B,N,D], got {tuple(latents.shape)}")
        batch, latent_len, dim = latents.shape
        if dim != self.latent_dim:
            raise ValueError(f"latent dim must be {self.latent_dim}, got {dim}")
        if latent_len > self.num_latents:
            raise ValueError(
                f"latent length {latent_len} exceeds num_latents={self.num_latents}"
            )
        target_len = self.max_len if seq_len is None else int(seq_len)
        if target_len > self.max_len:
            raise ValueError(
                f"sequence length {target_len} exceeds max_len={self.max_len}"
            )
        query = self.base_query.to(device=latents.device, dtype=latents.dtype)
        query = query.expand(batch, target_len, -1)
        mask = torch.ones(batch, target_len, device=latents.device, dtype=latents.dtype)
        x = self.query_gate(self.query_ffn(query), mask) + self.pos_emb(query)
        z = latents + self.pos_latent(latents)
        for layer in self.layers:
            x = layer(z, x, mask)
        return self.out_proj(x)
