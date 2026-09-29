from __future__ import annotations

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
        max_len: int = 128,
    ):
        super().__init__()
        latent_dim = int(latent_dim)
        self.max_len = int(max_len)

        self.pos_emb = AbsolutePositionalEmbedding(latent_dim, self.max_len)
        self.pos_latent = AbsolutePositionalEmbedding(latent_dim, int(num_latents))
        self.base_query = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.query_ffn = FeedForward(latent_dim, mult=mlp_ratio)
        self.query_gate = ScaleGate()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim=latent_dim,
                    latent_dim=latent_dim,
                    hidden=latent_dim,
                    heads=int(heads),
                    dim_head=latent_dim // int(heads),
                    mlp_ratio=float(mlp_ratio),
                )
                for _ in range(int(layers))
            ]
        )
        self.out_proj = FeatureBottleneckProjection(latent_dim, int(bottleneck_dim), int(feature_dim))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        query = self.base_query.to(device=latents.device, dtype=latents.dtype)
        query = query.expand(latents.shape[0], self.max_len, -1)
        x = self.query_gate(self.query_ffn(query)) + self.pos_emb(query)
        z = latents + self.pos_latent(latents)
        for layer in self.layers:
            x = layer(z, x)
        return self.out_proj(x)
