from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden_dim = int(int(dim) * float(mult))
        self.net = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, int(dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.scale = int(dim) ** -0.5
        self.weight = nn.Parameter(torch.zeros(int(max_len), int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[: x.shape[1]].to(device=x.device, dtype=x.dtype) * self.scale


class FeatureBottleneckProjection(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, output_dim: int):
        super().__init__()
        self.down = nn.Linear(int(input_dim), int(bottleneck_dim), bias=False)
        self.up = nn.Linear(int(bottleneck_dim), int(output_dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class ScaleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_mlp = nn.Linear(1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale_mlp(x.new_ones(x.shape[:-1] + (1,)))
        return x * (scale + 1.0)


class LatentAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        *,
        hidden: int,
        heads: int,
        dim_head: int,
        latents_first: bool,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.latents_first = bool(latents_first)

        self.norm_ctx = CodecRMSNorm(int(embedding_dim))
        self.norm_latents = CodecRMSNorm(int(latent_dim))
        self.to_q = nn.Linear(int(latent_dim), self.hidden, bias=False)
        self.to_kv_ctx = nn.Linear(int(embedding_dim), self.hidden * 2, bias=False)
        self.to_kv_latents = nn.Linear(int(latent_dim), self.hidden * 2, bias=False)
        self.q_norm = CodecRMSNorm(self.dim_head)
        self.k_norm = CodecRMSNorm(self.dim_head)
        self.out = nn.Linear(self.hidden, int(latent_dim), bias=False)

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
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch, query_len, self.hidden)
        return self.out(y)


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
    ):
        super().__init__()
        self.attn = LatentAttention(
            embedding_dim,
            latent_dim,
            hidden=hidden,
            heads=heads,
            dim_head=dim_head,
            latents_first=True,
        )
        self.ffn = FeedForward(latent_dim, mult=mlp_ratio)

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
    ):
        super().__init__()
        self.attn = LatentAttention(
            latent_dim,
            embedding_dim,
            hidden=hidden,
            heads=heads,
            dim_head=dim_head,
            latents_first=False,
        )
        self.attn_gate = ScaleGate()
        self.ffn = FeedForward(embedding_dim, mult=mlp_ratio)
        self.ffn_gate = ScaleGate()

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_gate(self.attn(latents, x))
        x = x + self.ffn_gate(self.ffn(x))
        return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class TextRotaryEmbeddingFast(nn.Module):
    """Rotary embedding of the latent positions; the prefix tokens are not rotated."""

    def __init__(self, dim: int, seq_len: int, prefix_len: int, theta: float = 10000.0):
        super().__init__()
        dim = int(dim)
        freqs = 1.0 / (float(theta) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        pos = torch.arange(int(seq_len), dtype=torch.float32)
        freqs = torch.einsum("n,f->nf", pos, freqs).repeat_interleave(2, dim=-1)
        ones = torch.ones((int(prefix_len), freqs.shape[-1]), dtype=torch.float32)
        self.register_buffer("freqs_cos", torch.cat([ones, torch.cos(freqs)], dim=0), persistent=False)
        self.register_buffer("freqs_sin", torch.cat([torch.zeros_like(ones), torch.sin(freqs)], dim=0), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        cos = self.freqs_cos[:seq_len].to(device=t.device, dtype=t.dtype)
        sin = self.freqs_sin[:seq_len].to(device=t.device, dtype=t.dtype)
        return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(hidden_size)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + self.eps).to(input_dtype)
        return self.weight.to(input_dtype) * (hidden_states * inv_std)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp_0 = nn.Linear(self.frequency_embedding_size, int(hidden_size))
        self.mlp_2 = nn.Linear(int(hidden_size), int(hidden_size))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].to(torch.float32) * freqs[None]
        t_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp_2(F.silu(self.mlp_0(t_emb)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(dim) // self.heads
        self.qkv = nn.Linear(int(dim), int(dim) * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(int(dim), int(dim), bias=True)

    def forward(self, x: torch.Tensor, rope: nn.Module) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, channels)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_eff = int(int(hidden_dim) * 2 / 3)
        self.w12 = nn.Linear(int(dim), 2 * hidden_eff, bias=True)
        self.w3 = nn.Linear(hidden_eff, int(dim), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(int(hidden_size))
        self.attn = Attention(int(hidden_size), int(heads))
        self.norm2 = RMSNorm(int(hidden_size))
        self.mlp = SwiGLU(int(hidden_size), int(int(hidden_size) * float(mlp_ratio)))

    def forward(self, x: torch.Tensor, rope: nn.Module) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(int(hidden_size))
        self.linear = nn.Linear(int(hidden_size), int(out_channels), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class BottleneckTextProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(int(input_dim), int(bottleneck_dim), bias=False)
        self.proj2 = nn.Linear(int(bottleneck_dim), int(hidden_size), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))
