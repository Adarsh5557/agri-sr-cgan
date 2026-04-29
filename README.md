# agri-sr-cgan
AgriVision Super-Resolution — cGAN for Crop Disease Image Reconstruction

# AgriVision Super-Resolution — cGAN for Crop Disease Image Reconstruction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project was built for a competitive Kaggle challenge in the domain of **Agri-Tech automated diagnostics**. In real-world deployments, drones and cheap mobile sensors photograph crop leaves to detect early signs of disease. However, due to hardware limitations, thermal sensor noise, and 2G cellular compression, images transmitted back to the central server are often heavily degraded and unusable for automated classification.

The goal of this project is to build a **Conditional Generative Adversarial Network (cGAN)** that performs **blind 4× Super-Resolution** — taking severely degraded **32×32 pixel** input images and reconstructing high-fidelity **128×128 pixel** outputs, faithfully restoring high-frequency biological textures such as leaf veins, chlorosis patterns, and necrotic lesions.

---

## Problem Statement

Low-resolution inputs in the dataset contain:
- Severe information loss from sensor hardware limitations
- Thermal sensor noise
- JPEG compression artifacts from 2G cellular transmission

An automated diagnostic system cannot reliably classify these degraded images. The model must **simultaneously denoise and upscale** without hallucinating false biological symptoms — a critical constraint given the downstream disease classification use case.

---

## Objective

> Reconstruct pristine 128×128 high-resolution crop leaf images from degraded 32×32 inputs using a fully from-scratch cGAN architecture, trained to be both visually realistic and spatially faithful.

---

## Architecture

### Generator — SRResNet with Bicubic Skip Connection

The generator is a custom **SRResNet** built entirely from scratch. Its key design decision is a **bicubic skip connection**:


This forces the network to learn residual high-frequency detail on top of a strong spatial baseline, directly improving MAE by ensuring the reconstructed image never drifts far from the original input structure.

| Component | Details |
|---|---|
| Head | Conv2d(3 → 64, 9×9) + PReLU |
| Residual Blocks | 16 × ResBlock(64ch): Conv → BN → PReLU → Conv → BN |
| Mid Conv | Conv2d(64 → 64) + BN |
| Upsampling | 2× PixelShuffle blocks (4× total upscale) |
| Tail | Conv2d(64 → 3, 9×9) |
| Skip | Bicubic interpolation (scale_factor=4) added to output |
| Output Activation | `torch.clamp(..., 0.0, 1.0)` |

**Total Generator Parameters:** ~1.5M

### Discriminator — Deep Convolutional Classifier

| Component | Details |
|---|---|
| Conv blocks | 8 blocks: stride-1/stride-2, channels 64 → 512, LeakyReLU(0.2) |
| Pooling | AdaptiveAvgPool2d(1) → Flatten |
| FC Head | Linear(512→1024) → LeakyReLU → Linear(1024→1) |

### Perceptual Loss — VGG19 Feature Matching

A VGG19 network (weights loaded from competition-provided `.pth`, not internet) is used to extract features up to layer 18 for perceptual loss computation during GAN fine-tuning.

---

## Training Strategy

Training is split into two stages:

### Stage 1 — L1-Only Pretraining (150 Epochs)
- **Loss:** Pure L1 (pixel-wise MAE)
- **Optimizer:** Adam (lr=3e-4, betas=(0.9, 0.999))
- **Scheduler:** CosineAnnealingLR (eta_min=1e-6)
- **Purpose:** Teach the generator spatial faithfulness before adversarial training destabilizes it

### Stage 2 — GAN Fine-Tuning (50 Epochs)
- **Generator Loss:**
  - L1 loss × 10.0 (dominant — protects MAE)
  - VGG Perceptual loss × 0.1 (texture realism)
  - Adversarial loss × 0.001 (keeps discriminator pressure low)
- **Discriminator Loss:** BCE on real/fake predictions
- **Optimizer (G):** Adam (lr=3e-5)
- **Optimizer (D):** Adam (lr=1e-4)

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch Size | 16 per GPU (32 on 2× T4) |
| Residual Blocks | 16 |
| Base Channels | 64 |
| Stage 1 Epochs | 150 |
| Stage 2 Epochs | 50 |
| Seed | 42 |

---

## Data Augmentation

During training, the following augmentations are applied consistently to both LR and HR image pairs:
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotation (k ∈ {0, 1, 2, 3})

---

## Inference — TTA-8 Ensemble

The final inference pipeline combines two techniques to significantly reduce MAE:

### Test Time Augmentation (TTA-8)
Each input image is run through 8 geometric variants (original, hflip, vflip, hflip+vflip, 4 rotations), and the model outputs are inverse-transformed and averaged. The original orientation is double-weighted:

```python
output = (sr_original * 2.0 + sr_hflip + sr_vflip + ... + sr_rot270) / 9.0
```

### Model Ensemble (70/30)
The final submission averages predictions from both trained checkpoints:
- `G_final.pth` — Stage 2 GAN fine-tuned model (weight: **0.7**)
- `G_stage1.pth` — Stage 1 L1-pretrained model (weight: **0.3**)

```python
final_sr = G_final_tta * 0.7 + G_stage1_tta * 0.3
```

This ensemble approach leverages the texture quality of the GAN model while retaining the spatial precision of the L1 model.

---

## Evaluation Metric

**Mean Absolute Error (MAE)** — pixel-by-pixel across all three RGB channels of the 128×128 output images.

MAE is used by the leaderboard specifically to penalize spatial inaccuracies. A realistic-looking bacterial spot placed even a few pixels away from the ground-truth position incurs a significant MAE penalty. This is why the bicubic skip connection, L1-dominant loss weighting, and TTA averaging are all critical design choices.

---

## Architectural Constraints (Competition Rules)

1. **No pretrained weights for the core task** — Generator and Discriminator are randomly initialized. VGG19 weights are loaded only for perceptual loss computation, provided as competition input data.
2. **Fully offline inference** — Final Kaggle notebook runs with Internet Access OFF and External Data DISALLOWED.

---

## Dataset

| Split | Directory | Size | Resolution |
|---|---|---|---|
| Training LR | `train_Low_Resolution/` | ~N images | 32×32 |
| Training HR | `train_High_Resolution/` | ~N images | 128×128 |
| Test LR | `test_Low_Resolution/` | 495 images | 32×32 |

---

## Submission Format

Each generated 128×128 image is flattened into **49,152 values** (128 × 128 × 3 RGB channels) and saved as a CSV:


Sanity checks applied before submission:
- Row count equals 495
- Each row contains exactly 49,152 pixel values
- All pixel values are in range [0, 255]

---

## Project Structure

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| PyTorch 2.x | Model building & training |
| torchvision | TF transforms, VGG19 structure |
| PIL / Pillow | Image loading |
| NumPy | Array ops, pixel flattening |
| Pandas | Submission CSV generation |
| tqdm | Training progress bars |
| Kaggle Notebooks | Training (2× T4 GPU) & offline inference |

---

## Competition Details

| Field | Info |
|---|---|
| Competition | Plant Leaves Super-Resolution Challenge |
| Platform | Kaggle |
| Duration | April 17, 2026 — April 21, 2026 |
| Metric | MAE (pixel-level) |
| Test Set | 495 images |

---

## Download Pretrained Weights
Download from [Releases](https://github.com/your-username/your-repo/releases/tag/v1.0):
- `G_final.pth` — Stage 2 GAN fine-tuned model
- `G_stage1.pth` — Stage 1 L1 pretrained model



## License

This project is for educational and competitive purposes. The dataset belongs to the competition organizers.










