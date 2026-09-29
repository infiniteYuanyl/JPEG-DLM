from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .layers import (
    BottleneckTextProjection,
    DiTBlock,
    FinalLayer,
    TextRotaryEmbeddingFast,
    TimestepEmbedder,
)


class LatentDiT(nn.Module):
    def __init__(
        self,
        z_dim: int = 512,
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        latent_len: int = 32,
        time_tokens: int = 4,
        sc_tokens: int = 4,
        mode_tokens: int = 4,
        freq_dim: int = 256,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden = int(hidden)
        self.prefix_len = int(time_tokens) + int(sc_tokens) + int(mode_tokens)

        self.self_cond_proj = nn.Linear(2 * self.z_dim, self.z_dim, bias=True)
        self.in_proj = BottleneckTextProjection(self.z_dim, self.hidden, int(bottleneck_dim))
        self.t_embedder = TimestepEmbedder(self.hidden, frequency_embedding_size=int(freq_dim))
        self.t_emb_tokens = nn.Parameter(torch.zeros(1, int(time_tokens), self.hidden))
        self.sc_embedder = TimestepEmbedder(self.hidden, frequency_embedding_size=int(freq_dim))
        self.sc_tokens = nn.Parameter(torch.zeros(1, int(sc_tokens), self.hidden))
        self.mode_tokens = nn.Parameter(torch.zeros(1, int(mode_tokens), self.hidden))
        self.rope = TextRotaryEmbeddingFast(
            dim=self.hidden // int(heads),
            seq_len=int(latent_len),
            prefix_len=self.prefix_len,
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(self.hidden, int(heads), mlp_ratio=float(mlp_ratio)) for _ in range(int(depth))]
        )
        self.denoise_head = FinalLayer(self.hidden, self.z_dim)
        self.decode_head = FinalLayer(self.hidden, self.z_dim)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        *,
        sc_scale: torch.Tensor,
        self_cond: Optional[torch.Tensor] = None,
        decode: bool = False,
    ) -> torch.Tensor:
        batch = z.shape[0]
        if self_cond is None:
            self_cond = torch.zeros_like(z)

        with torch.autocast(device_type=z.device.type, enabled=False):
            x = torch.cat([z, self_cond], dim=-1)
            x = self.self_cond_proj(x.float())
            x = self.in_proj(x.float())
            time_tok = self.t_emb_tokens.expand(batch, -1, -1) + self.t_embedder(t).unsqueeze(1)
            sc_tok = self.sc_tokens.expand(batch, -1, -1) + self.sc_embedder(sc_scale).unsqueeze(1)
            mode_tok = self.mode_tokens.expand(batch, -1, -1) * (1.0 if decode else 0.0)
            seq = torch.cat([time_tok, sc_tok, mode_tok, x], dim=1)

        for block in self.blocks:
            seq = block(seq, self.rope)

        out = seq[:, self.prefix_len :]
        with torch.autocast(device_type=z.device.type, enabled=False):
            head = self.decode_head if decode else self.denoise_head
            return head(out.float())
