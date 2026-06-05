# UniRestore-Unofficial

> Unofficial PyTorch implementation starter for **UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior** (CVPR 2025).
>
> If this repo saves you reading / reproduction time, please star it and follow [@StaryMoon](https://github.com/StaryMoon). I am building honest open reproduction starters for recent CVPR papers.

## Status

This repository is an **independent, unofficial, work-in-progress starter**.

- Paper: [UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html)
- Venue: CVPR 2025
- Reproduction status: **benchmarks are not reproduced yet**
- Relationship to authors: this repo is not official and is not affiliated with the paper authors.

## What Is Implemented

This v0.1.0 starter implements a compact, readable scaffold inspired by the paper:

- compact image encoder and decoder
- task/prompt conditioning tokens
- residual restoration head
- toy L1 + prompt-consistency loss
- smoke-test script

The goal is to make the high-level idea easy to inspect, fork, and improve.

## What Is Not Implemented Yet

- exact paper architecture
- official checkpoints
- dataset-specific training pipeline
- PSNR / SSIM benchmark reproduction

## Quick Start

```bash
git clone https://github.com/StaryMoon/UniRestore-Unofficial.git
cd UniRestore-Unofficial
pip install -r requirements.txt
python scripts/smoke_test.py
```

Expected output includes:

```text
loss: ...
restored: torch.Size([2, 3, 64, 64])
```

## Minimal Usage

```python
import torch
from unirestore_unofficial import UnofficialStarter

image = torch.rand(2, 3, 64, 64)
model = UnofficialStarter(kind="restoration")
out = model(image)
```

## Roadmap

- [ ] Replace toy modules with a closer implementation of the paper.
- [ ] Add dataset loader and config files.
- [ ] Add metric scripts and visualization.
- [ ] Reproduce a small benchmark or ablation table.
- [ ] Add pretrained weights once experiments are stable.

## Search Tags

cvpr-2025, image-restoration, diffusion-prior, pytorch, unofficial-implementation

## Citation

Please cite the original paper if you use the method. This repo is only an unofficial starter and does not replace the paper.

## License

MIT License. The original paper and official materials remain owned by their respective authors / publishers.
