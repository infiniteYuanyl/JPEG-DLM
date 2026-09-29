from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg
