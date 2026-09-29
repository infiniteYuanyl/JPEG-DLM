from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from .layers import (
    BottleneckTextProjection,
    DiTBlock,
    FinalLayer,
    TextRotaryEmbeddingFast,
    TimestepEmbedder,
    _xavier_zero_linear,
)


@contextmanager
def _autocast_disabled(device_type: str):
    with torch.amp.autocast(device_type=device_type, enabled=False):
        yield


class LatentDiT(nn.Module):
    def __init__(
        self,
        z_dim: int = 512,
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
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
        self.depth = int(depth)
        self.heads = int(heads)
        self.latent_len = int(latent_len)
        self.time_tokens = int(time_tokens)
        self.sc_tokens_count = int(sc_tokens)
        self.mode_tokens_count = int(mode_tokens)
        self.freq_dim = int(freq_dim)
        if self.hidden % self.heads != 0:
            raise ValueError("hidden must be divisible by heads")

        self.self_cond_proj = _xavier_zero_linear(2 * self.z_dim, self.z_dim, bias=True)
        self.in_proj = BottleneckTextProjection(
            self.z_dim,
            self.hidden,
            int(bottleneck_dim),
        )
        self.t_embedder = TimestepEmbedder(self.hidden, frequency_embedding_size=self.freq_dim)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, self.time_tokens, self.hidden))
        nn.init.normal_(self.t_emb_tokens, mean=0.0, std=0.02)
        self.sc_embedder = TimestepEmbedder(self.hidden, frequency_embedding_size=self.freq_dim)
        self.sc_tokens = nn.Parameter(torch.empty(1, self.sc_tokens_count, self.hidden))
        nn.init.normal_(self.sc_tokens, mean=0.0, std=0.02)
        self.mode_tokens = nn.Parameter(torch.empty(1, self.mode_tokens_count, self.hidden))
        nn.init.normal_(self.mode_tokens, mean=0.0, std=0.02)

        prefix = self.time_tokens + self.sc_tokens_count + self.mode_tokens_count
        self.rope = TextRotaryEmbeddingFast(
            dim=self.hidden // self.heads,
            pt_seq_len=self.latent_len,
            num_empty_token=prefix,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden,
                    self.heads,
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(self.depth)
            ]
        )
        self.denoise_head = FinalLayer(self.hidden, self.z_dim)
        self.decode_head = FinalLayer(self.hidden, self.z_dim)

    @staticmethod
    def _batch_vector(
        value,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)
        if value.ndim == 0 or value.numel() == 1:
            return value.reshape(1).expand(batch)
        value = value.reshape(value.size(0), -1)[:, 0]
        if value.size(0) != batch:
            raise ValueError(f"scalar condition has batch {value.size(0)}, expected {batch}")
        return value

    def forward(
        self,
        z: torch.Tensor,
        t,
        self_cond: Optional[torch.Tensor] = None,
        sc_scale=None,
        decode: bool = False,
    ) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"z must be [B,N,D], got {tuple(z.shape)}")
        batch, latent_len, z_dim = z.shape
        if z_dim != self.z_dim:
            raise ValueError(f"z dim must be {self.z_dim}, got {z_dim}")
        if latent_len > self.latent_len:
            raise ValueError(
                f"latent length {latent_len} exceeds configured length {self.latent_len}"
            )
        if self_cond is None:
            self_cond = torch.zeros_like(z)
        elif self_cond.shape != z.shape:
            raise ValueError(
                f"self_cond shape {tuple(self_cond.shape)} must match z {tuple(z.shape)}"
            )
        t_vec = self._batch_vector(t, batch, z.device, z.dtype)
        if sc_scale is None:
            sc_scale = 1.0
        sc_vec = self._batch_vector(sc_scale, batch, z.device, z.dtype)

        with _autocast_disabled(z.device.type):
            x = torch.cat([z, self_cond], dim=-1)
            x = self.self_cond_proj(x.float())
            x = self.in_proj(x.float())
            time = self.t_embedder(t_vec)
            time_tok = self.t_emb_tokens.expand(batch, -1, -1) + time.unsqueeze(1)
            sc = self.sc_embedder(sc_vec)
            sc_tok = self.sc_tokens.expand(batch, -1, -1) + sc.unsqueeze(1)
            mode_tok = self.mode_tokens.expand(batch, -1, -1)
            mode_tok = mode_tok * (1.0 if bool(decode) else 0.0)
            # Prefix order: time tokens, self-conditioning scale tokens, mode tokens, then latents.
            seq = torch.cat([time_tok, sc_tok, mode_tok, x], dim=1)

        deterministic = not (self.training and torch.is_grad_enabled())
        for block in self.blocks:
            seq = block(seq, self.rope, deterministic=deterministic)

        prefix = self.time_tokens + self.sc_tokens_count + self.mode_tokens_count
        out = seq[:, prefix:]
        with _autocast_disabled(z.device.type):
            head = self.decode_head if bool(decode) else self.denoise_head
            return head(out.float())
