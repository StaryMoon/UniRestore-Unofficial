from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from unirestore_unofficial import ModelConfig, UnofficialModel, reconstruction_loss


def main() -> None:
    torch.manual_seed(2026)
    config = ModelConfig(task="restoration", hidden_dim=64, num_layers=2, num_heads=4, output_dim=64)
    model = UnofficialModel(config)
    image = torch.rand(2, 3, 64, 64)
    condition = torch.randn(2, 4, config.hidden_dim)
    out = model(image, condition=condition)
    loss = reconstruction_loss(out.primary)
    loss.backward()
    print(f"output: {tuple(out.primary.shape)}")
    print(f"loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
