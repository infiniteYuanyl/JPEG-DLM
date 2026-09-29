from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from cores.modules.flow import euler_rollout, solver_grid

from .compressor import Compressor
from .decompressor import Decompressor
from .dit import LatentDiT


class JPEGDLM(nn.Module):
    """JPEG-DLM: compressor, latent DiT, decompressor and token projection.

    The compressor maps T5-small encoder features to a short latent sequence,
    the DiT generates latents with flow matching, and the decompressor and the
    token projection map latents back to tokens.
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg.model
        self.seq_len = int(cfg.data.seq_len)
        self.latent_len = int(model_cfg.latent_len)
        self.latent_dim = int(model_cfg.latent_dim)
        self.feature_dim = 512  # width of the T5-small features

        codec_cfg = model_cfg.codec
        self.compressor = Compressor(
            feature_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            bottleneck_dim=int(model_cfg.bottleneck_dim),
            num_latents=self.latent_len,
            layers=int(codec_cfg.layers),
            heads=int(codec_cfg.heads),
            mlp_ratio=float(codec_cfg.mlp_ratio),
            max_len=self.seq_len,
        )
        self.decompressor = Decompressor(
            latent_dim=self.latent_dim,
            feature_dim=self.feature_dim,
            bottleneck_dim=int(model_cfg.bottleneck_dim),
            num_latents=self.latent_len,
            layers=int(codec_cfg.layers),
            heads=int(codec_cfg.heads),
            mlp_ratio=float(codec_cfg.mlp_ratio),
            max_len=self.seq_len,
        )
        self.unembed_proj = nn.Linear(self.feature_dim, 512)
        self.lm_head = nn.Linear(512, int(model_cfg.vocab_size), bias=True)

        dit_cfg = model_cfg.dit
        self.dit = LatentDiT(
            z_dim=self.latent_dim,
            hidden=int(dit_cfg.hidden),
            depth=int(dit_cfg.depth),
            heads=int(dit_cfg.heads),
            mlp_ratio=float(dit_cfg.mlp_ratio),
            latent_len=self.latent_len,
            time_tokens=int(dit_cfg.time_tokens),
            sc_tokens=int(dit_cfg.sc_tokens),
            mode_tokens=int(dit_cfg.mode_tokens),
            freq_dim=int(dit_cfg.freq_dim),
            bottleneck_dim=int(model_cfg.bottleneck_dim),
        )
        # Per-channel mean and inverse standard deviation (1 / sigma) of the compressed latents.
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_scale", torch.ones(self.latent_dim))

    def standardize(self, z_raw: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.to(device=z_raw.device, dtype=z_raw.dtype).view(1, 1, -1)
        scale = self.latent_scale.to(device=z_raw.device, dtype=z_raw.dtype).view(1, 1, -1)
        return (z_raw - mean) * scale

    def unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.to(device=z.device, dtype=z.dtype).view(1, 1, -1)
        scale = self.latent_scale.to(device=z.device, dtype=z.dtype).view(1, 1, -1)
        return z / scale + mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Equal to unstandardize(z) up to float rounding; the paper's samples were drawn
        # with this exact sequence.
        z = self.unstandardize(self.standardize(self.unstandardize(z)))
        return self.decompressor(z)

    def logits(self, h_hat: torch.Tensor) -> torch.Tensor:
        return self.lm_head(F.gelu(self.unembed_proj(h_hat)))

    @classmethod
    def from_pretrained(cls, checkpoint, cfg, device="cpu"):
        """Build the model from a loaded configuration and load a ``.safetensors`` checkpoint."""
        model = cls(cfg)
        model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
        return model.to(torch.device(device)).eval()

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        *,
        steps: int = 32,
        sc_scale: float = 3.0,
        quantile_grid: bool = False,
    ) -> torch.Tensor:
        """Draw ``num_samples`` token rows with the Euler sampler.

        After the last Euler step, the decode branch predicts the clean
        embedding once at t = 1, and the decompressor and the token projection
        map it to token ids. ``quantile_grid`` uses the fixed quantile time grid
        of ``solver_grid`` instead of a random one.
        """
        count = int(num_samples)
        device = next(self.parameters()).device
        z = torch.randn(count, self.latent_len, self.latent_dim, device=device, dtype=torch.float32)
        scale = torch.full((count,), float(sc_scale), device=device, dtype=torch.float32)
        grid = solver_grid(
            int(steps),
            mean=-1.5,
            std=0.8,
            device=device,
            dtype=torch.float32,
            quantile=quantile_grid,
        )

        kernel_context = nullcontext()
        if device.type == "cuda":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # The samples in the paper were drawn with the memory-efficient attention kernel
            # (math is the fallback); flash attention rounds bf16 differently.
            kernel_context = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        with kernel_context:
            z = euler_rollout(self.dit, z, grid, sc_scale=scale)
            ones = torch.ones(count, device=device, dtype=torch.float32)
            x_dec = self.dit(z, ones, sc_scale=scale, decode=True)
            return self.logits(self.decode(x_dec)).argmax(dim=-1)
