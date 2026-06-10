from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    task: str = "vision"
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    output_dim: int = 64
    vocab_size: int = 32000


@dataclass
class ModelOutput:
    primary: torch.Tensor
    features: torch.Tensor


class TokenMixer(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_cond = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.norm_x(tokens), self.norm_cond(condition), self.norm_cond(condition))
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_mlp(tokens))


class UnofficialModel(nn.Module):
    """Compact PyTorch interface for unofficial paper reproduction work."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = ModelConfig(**kwargs)
        self.config = config
        hidden_dim = config.hidden_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8 if (hidden_dim // 2) % 8 == 0 else 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.SiLU(),
        )
        self.token_projection = nn.Linear(hidden_dim, hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TokenMixer(hidden_dim=hidden_dim, num_heads=config.num_heads)
            for _ in range(config.num_layers)
        ])
        self.pool = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, config.output_dim)
        self.image_head = nn.Conv2d(hidden_dim, 3, kernel_size=1)

    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        features = self.image_encoder(image)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return self.token_projection(tokens), (height, width)

    def forward(self, image: torch.Tensor, condition: torch.Tensor | None = None) -> ModelOutput:
        tokens, size = self.encode_image(image)
        if condition is None:
            condition = tokens.mean(dim=1, keepdim=True)
        condition = self.condition_projection(condition)
        for layer in self.layers:
            tokens = layer(tokens, condition)

        if self.config.task in {"restoration", "segmentation", "generation"}:
            fmap = tokens.transpose(1, 2).reshape(image.shape[0], self.config.hidden_dim, *size)
            pred = self.image_head(fmap)
            pred = F.interpolate(pred, size=image.shape[-2:], mode="bilinear", align_corners=False)
            if self.config.task == "restoration":
                pred = (image + pred).clamp(0, 1)
            return ModelOutput(primary=pred, features=tokens)

        pooled = self.pool(tokens.mean(dim=1))
        return ModelOutput(primary=self.head(pooled), features=tokens)


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    if target is None or target.shape != prediction.shape:
        target = torch.zeros_like(prediction)
    return F.smooth_l1_loss(prediction, target)
