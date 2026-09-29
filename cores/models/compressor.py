from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    AbsolutePositionalEmbedding,
    CodecRMSNorm,
    EncoderBlock,
    FeatureBottleneckProjection,
)


class Compressor(nn.Module):
    """Compress token-length features into the latent sequence."""

    def __init__(
        self,
        feature_dim: int = 512,
        latent_dim: int = 512,
        bottleneck_dim: int = 128,
        num_latents: int = 32,
        layers: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        max_len: int = 128,
    ):
        super().__init__()
        latent_dim = int(latent_dim)

        self.in_proj = FeatureBottleneckProjection(int(feature_dim), int(bottleneck_dim), latent_dim)
        self.latents = nn.Parameter(torch.zeros(int(num_latents), latent_dim))
        self.latent_norm = CodecRMSNorm(latent_dim)
        self.pos_emb = AbsolutePositionalEmbedding(latent_dim, int(max_len))
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
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
        self.output_norm = CodecRMSNorm(latent_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        ctx = self.in_proj(hidden)
        ctx = ctx + self.pos_emb(ctx)
        latents = self.latents.to(device=hidden.device, dtype=hidden.dtype)
        latents = latents.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        latents = self.latent_norm(latents)
        for layer in self.layers:
            latents = layer(ctx, latents)
        return self.output_norm(latents)
