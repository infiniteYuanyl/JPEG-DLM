from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_zero_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class CodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-8):
        super().__init__()
        self.scale = int(dim) ** -0.5
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(int(dim) * float(mult))
        self.net = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.scale = int(dim) ** -0.5
        self.max_len = int(max_len)
        self.weight = nn.Parameter(torch.empty(self.max_len, int(dim)))
        # Same initialization as nn.Embedding: N(0, 1).
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds positional table: "
                f"{seq_len} > {self.max_len}"
            )
        return self.weight[:seq_len].to(device=x.device, dtype=x.dtype) * self.scale


class FeatureBottleneckProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        self.down = nn.Linear(int(input_dim), int(bottleneck_dim), bias=bias)
        self.up = nn.Linear(int(bottleneck_dim), int(output_dim), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class ScaleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_mlp = nn.Linear(1, 1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scale = self.scale_mlp(mask.unsqueeze(-1))
        return x * (scale + 1.0)


class LatentAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        *,
        hidden: int = 512,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        latents_first: bool = True,
    ):
        super().__init__()
        hidden = int(hidden)
        heads = int(heads)
        dim_head = int(dim_head)
        if hidden != heads * dim_head:
            raise ValueError(
                f"hidden must equal heads * dim_head, got {hidden} != {heads} * {dim_head}"
            )
        self.hidden = hidden
        self.heads = heads
        self.dim_head = dim_head
        self.latents_first = bool(latents_first)
        self.dropout = float(dropout)

        self.norm_ctx = CodecRMSNorm(int(embedding_dim), eps=1.0e-8)
        self.norm_latents = CodecRMSNorm(int(latent_dim), eps=1.0e-8)
        self.to_q = nn.Linear(int(latent_dim), hidden, bias=False)
        self.to_kv_ctx = nn.Linear(int(embedding_dim), hidden * 2, bias=False)
        self.to_kv_latents = nn.Linear(int(latent_dim), hidden * 2, bias=False)
        self.q_norm = CodecRMSNorm(dim_head, eps=1.0e-8)
        self.k_norm = CodecRMSNorm(dim_head, eps=1.0e-8)
        self.out = nn.Linear(hidden, int(latent_dim), bias=False)
        self.proj_dropout = nn.Dropout(self.dropout)

    def forward(self, ctx: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        batch, query_len, _ = latents.shape
        ctx = self.norm_ctx(ctx)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        kv_ctx = self.to_kv_ctx(ctx)
        kv_latents = self.to_kv_latents(latents)
        if self.latents_first:
            kv = torch.cat((kv_latents, kv_ctx), dim=1)
        else:
            kv = torch.cat((kv_ctx, kv_latents), dim=1)
        k, v = kv.split(self.hidden, dim=-1)

        q = q.view(batch, query_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch, k.shape[1], self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, v.shape[1], self.heads, self.dim_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch, query_len, self.hidden)
        return self.proj_dropout(self.out(y))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        latent_dim: int,
        hidden: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.attn = LatentAttention(
            embedding_dim,
            latent_dim,
            hidden=hidden,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            latents_first=True,
        )
        self.ffn = FeedForward(latent_dim, mult=mlp_ratio, dropout=dropout)

    def forward(self, ctx: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        latents = latents + self.attn(ctx, latents)
        latents = latents + self.ffn(latents)
        return latents


class DecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        latent_dim: int,
        hidden: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.attn = LatentAttention(
            latent_dim,
            embedding_dim,
            hidden=hidden,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            latents_first=False,
        )
        self.attn_gate = ScaleGate()
        self.ffn = FeedForward(embedding_dim, mult=mlp_ratio, dropout=dropout)
        self.ffn_gate = ScaleGate()

    def forward(
        self,
        latents: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn_gate(self.attn(latents, x), mask)
        x = x + self.ffn_gate(self.ffn(x), mask)
        return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class TextRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 32,
        theta: float = 10000.0,
        num_empty_token: int = 12,
    ):
        super().__init__()
        self.dim = int(dim)
        self.pt_seq_len = int(pt_seq_len)
        self.theta = float(theta)
        self.num_empty_token = int(num_empty_token)
        freqs_cos, freqs_sin = self._compute_freqs()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _compute_freqs(self) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32)[: self.dim // 2]
                / self.dim
            )
        )
        pos = torch.arange(self.pt_seq_len, dtype=torch.float32)
        freqs_main = torch.einsum("n,f->nf", pos, freqs)
        freqs_main = freqs_main.repeat_interleave(2, dim=-1)
        cos_parts = []
        sin_parts = []
        if self.num_empty_token > 0:
            cos_parts.append(
                torch.ones((self.num_empty_token, freqs_main.shape[-1]), dtype=torch.float32)
            )
            sin_parts.append(
                torch.zeros((self.num_empty_token, freqs_main.shape[-1]), dtype=torch.float32)
            )
        cos_parts.append(torch.cos(freqs_main))
        sin_parts.append(torch.sin(freqs_main))
        return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        cos = self.freqs_cos[:seq_len].to(device=t.device, dtype=t.dtype)
        sin = self.freqs_sin[:seq_len].to(device=t.device, dtype=t.dtype)
        return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + self.eps).to(input_dtype)
        return self.weight.to(input_dtype) * (hidden_states * inv_std)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp_0 = nn.Linear(self.frequency_embedding_size, self.hidden_size)
        self.mlp_2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.normal_(self.mlp_0.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlp_0.bias)
        nn.init.normal_(self.mlp_2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlp_2.bias)

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = int(dim) // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].to(torch.float32) * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if int(dim) % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp_2(F.silu(self.mlp_0(t_emb)))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim = int(dim)
        heads = int(heads)
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.qkv = _xavier_zero_linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=1.0e-6) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=1.0e-6) if qk_norm else nn.Identity()
        self.proj = _xavier_zero_linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[nn.Module],
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if not deterministic else 0.0,
            is_causal=False,
        )
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, channels)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        hidden_eff = int(int(hidden_dim) * 2 / 3)
        self.dropout = float(dropout)
        self.w12 = _xavier_zero_linear(int(dim), 2 * hidden_eff, bias=True)
        self.w3 = _xavier_zero_linear(hidden_eff, int(dim), bias=True)

    def forward(self, x: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        if self.dropout > 0.0:
            hidden = F.dropout(hidden, p=self.dropout, training=not deterministic)
        return self.w3(hidden)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(int(hidden_size), eps=1.0e-6)
        self.attn = Attention(
            int(hidden_size),
            heads=int(heads),
            qkv_bias=True,
            qk_norm=True,
            dropout=float(dropout),
        )
        self.norm2 = RMSNorm(int(hidden_size), eps=1.0e-6)
        self.mlp = SwiGLU(
            int(hidden_size),
            int(int(hidden_size) * float(mlp_ratio)),
            dropout=float(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[nn.Module],
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, deterministic=deterministic)
        x = x + self.mlp(self.norm2(x), deterministic=deterministic)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(int(hidden_size), eps=1.0e-6)
        self.linear = nn.Linear(int(hidden_size), int(out_channels), bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class BottleneckTextProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        if self.bottleneck_dim <= 0:
            self.proj = (
                nn.Identity()
                if self.input_dim == self.hidden_size
                else _xavier_zero_linear(self.input_dim, self.hidden_size, bias=True)
            )
        else:
            self.proj1 = _xavier_zero_linear(
                self.input_dim,
                self.bottleneck_dim,
                bias=False,
            )
            self.proj2 = _xavier_zero_linear(
                self.bottleneck_dim,
                self.hidden_size,
                bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bottleneck_dim <= 0:
            return self.proj(x)
        return self.proj2(self.proj1(x))
