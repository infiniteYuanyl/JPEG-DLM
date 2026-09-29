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
        dropout: float = 0.0,
        max_len: int = 128,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_latents = int(num_latents)
        self.max_len = int(max_len)
        if self.latent_dim % int(heads) != 0:
            raise ValueError("latent_dim must be divisible by heads")
        dim_head = self.latent_dim // int(heads)

        self.in_proj = FeatureBottleneckProjection(
            self.feature_dim,
            self.bottleneck_dim,
            self.latent_dim,
            bias=False,
        )
        self.latents = nn.Parameter(torch.empty(self.num_latents, self.latent_dim))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        self.latent_norm = CodecRMSNorm(self.latent_dim, eps=1.0e-8)
        self.pos_emb = AbsolutePositionalEmbedding(self.latent_dim, self.max_len)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
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
        self.output_norm = CodecRMSNorm(self.latent_dim, eps=1.0e-8)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be [B,L,F], got {tuple(hidden.shape)}")
        batch, seq_len, feature_dim = hidden.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"hidden feature dim must be {self.feature_dim}, got {feature_dim}"
            )
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len={self.max_len}")
        ctx = self.in_proj(hidden)
        ctx = ctx + self.pos_emb(ctx)
        latents = self.latents.to(device=hidden.device, dtype=hidden.dtype)
        latents = latents.unsqueeze(0).expand(batch, -1, -1)
        latents = self.latent_norm(latents)
        for layer in self.layers:
            latents = layer(ctx, latents)
        return self.output_norm(latents)
