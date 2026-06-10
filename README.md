# UniRestore-Unofficial

<div align="center">

**Unofficial PyTorch Reproduction of**  
# UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior

[CVPR 2025]  
![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Unofficial-Reproduction-orange)

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html) · [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.pdf) · [Issues](https://github.com/StaryMoon/UniRestore-Unofficial/issues) · [Release](https://github.com/StaryMoon/UniRestore-Unofficial/releases)

</div>

> This is an **unofficial** implementation maintained by [@StaryMoon](https://github.com/StaryMoon). If this repository helps your reading, reproduction, or course project, please consider giving it a star and following my GitHub profile.

## News

- **2026-06-10**: Repository upgraded with an official-style README, paper citation metadata, cleaner package interfaces, default configuration, and release-ready project structure.

## Overview

This repository organizes a PyTorch implementation for **UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior**, focusing on unified perceptual and task-oriented restoration with diffusion priors. The codebase is structured like a standard research repository so that model components, configuration files, scripts, and evaluation utilities can be extended independently.

Main goals:

- provide a clean PyTorch module layout for the paper;
- keep training, inference, evaluation, and configuration entry points explicit;
- track paper-reported metrics separately from local experiment logs;
- make it easy for contributors to inspect, compare, and extend the implementation.

## Repository Structure

```text
UniRestore-Unofficial/
├── configs/
│   └── default.yaml
├── scripts/
│   └── smoke_test.py
├── src/unirestore_unofficial/
│   ├── __init__.py
│   └── model.py
├── CITATION.cff
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/StaryMoon/UniRestore-Unofficial.git
cd UniRestore-Unofficial
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled experiments, install the PyTorch build matching your CUDA version from the official PyTorch website before installing the rest of the dependencies.

## Quick Check

Run the minimal forward-pass check:

```bash
python scripts/smoke_test.py
```

Expected output:

```text
output: (...)
loss: ...
```

This confirms that the package import path, model interface, and tensor flow are working.

## Data Preparation

Create local data folders:

```bash
mkdir -p data/train data/val data/test checkpoints outputs
```

Recommended layout:

```text
data/
├── train/
├── val/
└── test/
```

Keep private datasets, downloaded checkpoints, and generated outputs out of git. Dataset-specific converters can be added under `scripts/` while preserving the public repository structure.

## Training

Minimal module usage:

```python
import torch
from unirestore_unofficial import ModelConfig, UnofficialModel, reconstruction_loss

config = ModelConfig(task="restoration", hidden_dim=128, num_layers=2, num_heads=4)
model = UnofficialModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(2, 3, 64, 64)
condition = torch.randn(2, 4, config.hidden_dim)
target = torch.zeros(2, config.output_dim)

out = model(x, condition=condition)
loss = reconstruction_loss(out.primary, target)
loss.backward()
optimizer.step()
```

The repository separates model code, configuration, experiment outputs, and evaluation logs so new components can be added without changing the public interface.

## Inference

```python
import torch
from unirestore_unofficial import UnofficialModel

model = UnofficialModel().eval()
with torch.no_grad():
    x = torch.randn(1, 3, 64, 64)
    y = model(x).primary
print(y.shape)
```

## Evaluation

Suggested entry points:

```bash
python scripts/smoke_test.py
# python scripts/evaluate.py --config configs/default.yaml --ckpt checkpoints/model.pt
```

Paper-reported values and local run values should be kept in separate columns so readers can distinguish citation numbers from local experiment logs.

## Paper Results

For copyright and license clarity, this repository links to the original paper figures and tables instead of redistributing screenshots copied from the PDF. The table below tracks where readers can find the paper-reported results.

| Result Type | Paper Location | Source |
|---|---|---|
| Main quantitative comparison | Main paper tables, pp. 17969-17979 | [CVF paper page](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html) |
| Ablation study | Ablation / experiment section | [CVF paper page](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html) |
| Qualitative examples | Main paper figures and supplemental material | [CVF paper page](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html) |

## Reproduction Log

| Date | Config | Split | Metric | Value | Notes |
|---|---|---|---:|---:|---|
| 2026-06-10 | `configs/default.yaml` | smoke check | forward pass | ok | package interface validation |

## Implementation Status

- [x] Package layout and install metadata
- [x] Core PyTorch module interfaces
- [x] Default config and smoke test
- [x] Paper citation and result-location index
- [ ] Dataset-specific preprocessing scripts
- [ ] Paper-specific training recipe
- [ ] Evaluation and visualization scripts
- [ ] Public checkpoints and model zoo entries

## Model Zoo

| Model | Checkpoint | Config | Notes |
|---|---|---|---|
| default | TBA | `configs/default.yaml` | compact implementation interface |

## Citation

If you find this repository useful, please cite the original paper:

```bibtex
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, I-Hsiang and Chen, Wei-Ting and Liu, Yu-Wei and Chiang, Yuan-Chun and Kuo, Sy-Yen and Yang, Ming-Hsuan},
    title     = {UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17969-17979}
}
```

## Acknowledgements

- Thanks to the authors of **UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior** for the original research.
- Thanks to the Computer Vision Foundation for maintaining the CVPR Open Access pages.
- This repository is inspired by standard open-source PyTorch research codebases.
- The implementation is unofficial and all paper names, datasets, and trademarks belong to their respective owners.

## License

This repository is released under the MIT License. The original paper, datasets, official code, project assets, and third-party dependencies remain governed by their own licenses.

## Keywords

cvpr-2025, pytorch, unofficial-implementation, image-restoration, diffusion-prior, low-level-vision
