from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class StarterOutput:
    primary: torch.Tensor
    aux: torch.Tensor


class TokenMixer(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm_x = nn.LayerNorm(embed_dim)
        self.norm_cond = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.norm_x(tokens), self.norm_cond(cond), self.norm_cond(cond))
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_mlp(tokens))


class UnofficialStarter(nn.Module):
    """Small readable scaffold inspired by the paper, not an exact reproduction."""

    def __init__(self, kind: str = "restoration", embed_dim: int = 64, num_tokens: int = 16):
        super().__init__()
        self.kind = kind
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8 if (embed_dim // 2) % 8 == 0 else 1, embed_dim // 2),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8 if embed_dim % 8 == 0 else 1, embed_dim),
            nn.SiLU(),
        )
        self.token_proj = nn.Linear(embed_dim, embed_dim)
        self.cond_proj = nn.Linear(embed_dim, embed_dim)
        self.mixer = TokenMixer(embed_dim=embed_dim)
        self.restoration_head = nn.Conv2d(embed_dim, 3, kernel_size=1)
        self.gaussian_head = nn.Linear(embed_dim, 11)
        self.video_head = nn.Linear(embed_dim, embed_dim)
        self.vlm_head = nn.Linear(embed_dim, 32)
        self.robot_head = nn.Linear(embed_dim, 7)

    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        fmap = self.image_encoder(image)
        height, width = fmap.shape[-2:]
        tokens = fmap.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        return tokens, (height, width)

    def forward(self, image: torch.Tensor, condition: torch.Tensor | None = None) -> StarterOutput:
        tokens, size = self.encode_image(image)
        if condition is None:
            condition = tokens.mean(dim=1, keepdim=True)
        condition = self.cond_proj(condition)
        tokens = self.mixer(tokens, condition)

        if self.kind == "restoration":
            fmap = tokens.transpose(1, 2).reshape(image.shape[0], self.embed_dim, *size)
            residual = F.interpolate(self.restoration_head(fmap), size=image.shape[-2:], mode="bilinear", align_corners=False)
            restored = (image + residual).clamp(0, 1)
            return StarterOutput(primary=restored, aux=tokens)
        if self.kind == "gaussian":
            pooled = tokens[:, :128]
            if pooled.shape[1] < 128:
                repeat = 128 // pooled.shape[1] + 1
                pooled = pooled.repeat(1, repeat, 1)[:, :128]
            gaussians = self.gaussian_head(pooled)
            return StarterOutput(primary=gaussians, aux=tokens)
        if self.kind == "video":
            video = self.video_head(tokens[:, :16]).reshape(image.shape[0], 8, 2, self.embed_dim)
            video = video.reshape(image.shape[0], 8, 16, self.embed_dim // 8)
            return StarterOutput(primary=video, aux=tokens)
        if self.kind == "robot":
            actions = self.robot_head(tokens[:, :8])
            return StarterOutput(primary=actions, aux=tokens)
        logits = self.vlm_head(tokens[:, :8])
        return StarterOutput(primary=logits, aux=tokens)

    def toy_loss(self, image: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        out = self.forward(image).primary
        if target is None:
            target = torch.zeros_like(out)
        if target.shape != out.shape:
            target = torch.zeros_like(out)
        return F.smooth_l1_loss(out, target)
