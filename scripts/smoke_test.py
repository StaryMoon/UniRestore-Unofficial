from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from unirestore_unofficial import UnofficialStarter


def main() -> None:
    torch.manual_seed(2026)
    image = torch.rand(2, 3, 64, 64)
    condition = torch.randn(2, 4, 64)
    model = UnofficialStarter(kind="restoration", embed_dim=64)
    out = model(image, condition)
    loss = model.toy_loss(image)
    loss.backward()
    print(f"loss: {loss.item():.6f}")
    print("restoration:", out.primary.shape)


if __name__ == "__main__":
    main()
