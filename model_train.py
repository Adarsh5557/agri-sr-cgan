#!/usr/bin/env python
# coding: utf-8

"""
AgriVision Super-Resolution — cGAN for Crop Disease Image Reconstruction
=========================================================================
Blind 4x Super-Resolution: 32x32 → 128x128
Architecture : SRResNet Generator + Deep CNN Discriminator + VGG19 Perceptual Loss
Inference    : TTA-8 Ensemble (G_final × 0.7 + G_stage1 × 0.3)
Platform     : Kaggle (2× T4 GPU, Internet OFF)
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ── REPRODUCIBILITY & DEVICE ─────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
print(f"Device : {DEVICE}")
print(f"GPUs   : {NUM_GPUS}")


# ── PATHS & HYPERPARAMETERS ──────────────────────────────────────────────────

BASE     = '/kaggle/input/competitions/plant-leaves-super-resolution-challenge'
HR_DIR   = f'{BASE}/train_High_Resolution'
LR_DIR   = f'{BASE}/train_Low_Resolution'
TEST_DIR = f'{BASE}/test_Low_Resolution'
VGG_PATH = f'{BASE}/vgg19_weights.pth'
WORK_DIR = '/kaggle/working'

BATCH      = 16 * max(NUM_GPUS, 1)   # 16 per GPU → 32 on 2×T4
EPOCHS_L1  = 150                      # Stage 1: L1 pretraining
EPOCHS_GAN = 50                       # Stage 2: GAN fine-tune
LR_G       = 3e-4
LR_D       = 1e-4
NUM_RES    = 16                       # Number of residual blocks
CH         = 64                       # Base channel width


# ── DATASET ──────────────────────────────────────────────────────────────────

class LeafSRDataset(Dataset):
    """
    Paired LR / HR dataset for crop leaf super-resolution.
    Supports data augmentation (hflip, vflip, rot90) for training.
    In inference mode (hr_dir=None), returns (lr_tensor, filename).
    """
    def __init__(self, lr_dir, hr_dir=None, augment=False):
        self.lr_dir  = lr_dir
        self.hr_dir  = hr_dir
        self.augment = augment
        self.lr_files = sorted(f for f in os.listdir(lr_dir) if f.endswith('.png'))
        if hr_dir:
            self.hr_files = sorted(f for f in os.listdir(hr_dir) if f.endswith('.png'))
            assert len(self.lr_files) == len(self.hr_files), \
                f"LR/HR mismatch: {len(self.lr_files)} vs {len(self.hr_files)}"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_t = TF.to_tensor(
            Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        )

        if self.hr_dir:
            hr_t = TF.to_tensor(
                Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
            )
            if self.augment:
                if random.random() > 0.5:
                    lr_t = TF.hflip(lr_t); hr_t = TF.hflip(hr_t)
                if random.random() > 0.5:
                    lr_t = TF.vflip(lr_t); hr_t = TF.vflip(hr_t)
                k = random.randint(0, 3)
                if k:
                    lr_t = torch.rot90(lr_t, k, [1, 2])
                    hr_t = torch.rot90(hr_t, k, [1, 2])
            return lr_t, hr_t

        return lr_t, self.lr_files[idx]   # inference mode


# ── GENERATOR ────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block: Conv → BN → PReLU → Conv → BN + skip."""
    def __init__(self, ch=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
    def forward(self, x):
        return x + self.body(x)


class UpBlock(nn.Module):
    """Sub-pixel convolution upsampling block (×2 per block)."""
    def __init__(self, ch=64):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
    def forward(self, x):
        return self.up(x)


class Generator(nn.Module):
    """
    SRResNet with bicubic skip connection.
    output = bicubic_upscale(LR) + residual_network(LR)

    The bicubic skip ensures spatial faithfulness and directly lowers MAE
    by anchoring the output to the input structure before residual learning.
    4× upscale: 32×32 → 128×128 via two ×2 PixelShuffle blocks.
    """
    def __init__(self, num_res=16, ch=64):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, ch, 9, 1, 4), nn.PReLU())
        self.res  = nn.Sequential(*[ResBlock(ch) for _ in range(num_res)])
        self.mid  = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.up   = nn.Sequential(UpBlock(ch), UpBlock(ch))   # 4× total
        self.tail = nn.Conv2d(ch, 3, 9, 1, 4)

    def forward(self, lr):
        bicubic = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
        h = self.head(lr)
        h = h + self.mid(self.res(h))
        h = self.tail(self.up(h))
        return torch.clamp(h + bicubic, 0.0, 1.0)


# ── DISCRIMINATOR ────────────────────────────────────────────────────────────

def _disc_block(in_ch, out_ch, stride=1, bn=True):
    """Helper to build a discriminator conv block."""
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """
    Deep convolutional discriminator.
    8 conv blocks (stride-1 / stride-2 alternating) → AdaptiveAvgPool → FC head.
    Channels: 3 → 64 → 128 → 256 → 512.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _disc_block(3,   64,  1, bn=False),
            _disc_block(64,  64,  2),
            _disc_block(64,  128, 1),
            _disc_block(128, 128, 2),
            _disc_block(128, 256, 1),
            _disc_block(256, 256, 2),
            _disc_block(256, 512, 1),
            _disc_block(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )
    def forward(self, x):
        return self.net(x)


# ── VGG19 PERCEPTUAL LOSS ─────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Feature-matching loss using VGG19 up to layer 18.
    Weights loaded from competition-provided .pth (no internet required).
    Frozen during training.
    """
    def __init__(self, vgg_path):
        super().__init__()
        from torchvision.models import vgg19
        vgg = vgg19(pretrained=False)
        vgg.load_state_dict(torch.load(vgg_path, map_location='cpu'))
        self.feat = nn.Sequential(*list(vgg.features)[:18]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        return F.l1_loss(self.feat(sr), self.feat(hr))


# ── UTILITY ──────────────────────────────────────────────────────────────────

def unwrap(model):
    """Unwrap DataParallel wrapper to access the underlying module."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ── STAGE 1: L1-ONLY PRETRAINING ─────────────────────────────────────────────

def train_stage1(G, loader, epochs, device):
    """
    Stage 1: Train generator with pure L1 loss.
    Teaches spatial faithfulness before adversarial training.
    Optimizer : Adam (lr=3e-4)
    Scheduler : CosineAnnealingLR
    """
    opt = Adam(unwrap(G).parameters(), lr=LR_G, betas=(0.9, 0.999))
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    G.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for lr_t, hr_t in tqdm(loader, desc=f'[L1] {epoch}/{epochs}', leave=False):
            lr_t, hr_t = lr_t.to(device), hr_t.to(device)
            loss = F.l1_loss(G(lr_t), hr_t)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        sch.step()
        if epoch % 10 == 0:
            avg = total / len(loader)
            print(f'  Epoch {epoch:3d} | L1={avg:.5f} | ~pixel MAE≈{avg*255:.2f}')
    return G


# ── STAGE 2: GAN FINE-TUNING ─────────────────────────────────────────────────

def train_stage2(G, D, loader, vgg_loss, epochs, device):
    """
    Stage 2: GAN fine-tuning with combined loss.
    Generator loss = L1 × 10.0 + VGG × 0.1 + Adversarial × 0.001
    L1 is kept dominant to protect MAE while GAN adds texture realism.
    """
    opt_g = Adam(unwrap(G).parameters(), lr=LR_G * 0.1, betas=(0.9, 0.999))
    opt_d = Adam(unwrap(D).parameters(), lr=LR_D,        betas=(0.9, 0.999))
    bce   = nn.BCEWithLogitsLoss()

    G.train(); D.train()
    for epoch in range(1, epochs + 1):
        g_sum = d_sum = 0.0
        for lr_t, hr_t in tqdm(loader, desc=f'[GAN] {epoch}/{epochs}', leave=False):
            lr_t, hr_t = lr_t.to(device), hr_t.to(device)
            sr = G(lr_t)

            # ── Discriminator update ──
            real_pred = D(hr_t)
            fake_pred = D(sr.detach())
            d_loss = 0.5 * (bce(real_pred, torch.ones_like(real_pred)) +
                            bce(fake_pred, torch.zeros_like(fake_pred)))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # ── Generator update (L1 dominates to protect MAE) ──
            g_loss = (F.l1_loss(sr, hr_t)                      * 10.0  +
                      vgg_loss(sr, hr_t)                         *  0.1  +
                      bce(D(sr), torch.ones_like(D(sr)))         * 0.001)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            g_sum += g_loss.item(); d_sum += d_loss.item()

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | G={g_sum/len(loader):.4f} '
                  f'| D={d_sum/len(loader):.4f}')
    return G


# ── BUILD MODELS ─────────────────────────────────────────────────────────────

train_ds = LeafSRDataset(LR_DIR, HR_DIR, augment=True)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                      num_workers=2, pin_memory=True)
print(f"Training samples : {len(train_ds)}  |  Batch size : {BATCH}")

G = Generator(num_res=NUM_RES, ch=CH).to(DEVICE)
D = Discriminator().to(DEVICE)

if NUM_GPUS > 1:
    print(f"Wrapping models with DataParallel across {NUM_GPUS} GPUs")
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

print(f"Generator params : {sum(p.numel() for p in unwrap(G).parameters()):,}")


# ── TRAINING ─────────────────────────────────────────────────────────────────

# Stage 1
print("\n=== Stage 1 : L1-only training ===")
G = train_stage1(G, train_dl, EPOCHS_L1, DEVICE)
torch.save(unwrap(G).state_dict(), f'{WORK_DIR}/G_stage1.pth')
print("Stage 1 done")

# Stage 2
print("\n=== Stage 2 : GAN fine-tune ===")
vgg_loss = VGGPerceptualLoss(VGG_PATH).to(DEVICE)
G = train_stage2(G, D, train_dl, vgg_loss, EPOCHS_GAN, DEVICE)
torch.save(unwrap(G).state_dict(), f'{WORK_DIR}/G_final.pth')
print("Stage 2 done")


# ── INFERENCE — TTA-8 ENSEMBLE ───────────────────────────────────────────────

test_ds = LeafSRDataset(TEST_DIR)
test_dl = DataLoader(test_ds, batch_size=8, shuffle=False,
                     num_workers=2, pin_memory=True)
print(f"\nTest samples: {len(test_ds)}")

# Load both checkpoints
G1 = Generator(num_res=NUM_RES, ch=CH).to(DEVICE)
G1.load_state_dict(torch.load(f'{WORK_DIR}/G_final.pth', map_location=DEVICE))
G1.eval()

G2 = Generator(num_res=NUM_RES, ch=CH).to(DEVICE)
G2.load_state_dict(torch.load(f'{WORK_DIR}/G_stage1.pth', map_location=DEVICE))
G2.eval()


def tta8(model, x):
    """
    Test Time Augmentation with 8 geometric variants.
    Original is double-weighted; all others weighted equally.
    Returns: averaged prediction in original orientation.
    """
    sr1 = model(x)                                                               # original (×2 weight)
    sr2 = torch.flip(model(torch.flip(x, [3])), [3])                            # hflip
    sr3 = torch.flip(model(torch.flip(x, [2])), [2])                            # vflip
    sr4 = torch.flip(model(torch.flip(x, [2, 3])), [2, 3])                      # hflip + vflip
    sr5 = torch.rot90(model(torch.rot90(x, 1, [2, 3])), 3, [2, 3])             # rot90
    sr6 = torch.rot90(model(torch.rot90(x, 2, [2, 3])), 2, [2, 3])             # rot180
    sr7 = torch.rot90(model(torch.rot90(x, 3, [2, 3])), 1, [2, 3])             # rot270
    sr8 = torch.flip(
            torch.rot90(
                model(torch.rot90(torch.flip(x, [3]), 1, [2, 3])),
            3, [2, 3]),
          [3])                                                                    # hflip + rot90
    return (sr1 * 2.0 + sr2 + sr3 + sr4 + sr5 + sr6 + sr7 + sr8) / 9.0


rows = []
with torch.no_grad():
    for lr_t, fnames in tqdm(test_dl, desc='TTA-8 Ensemble Inference'):
        lr_t = lr_t.to(DEVICE)

        out1 = tta8(G1, lr_t)             # G_final  with TTA
        out2 = tta8(G2, lr_t)             # G_stage1 with TTA

        # Weighted ensemble: trust G_final more (70/30)
        sr = (out1 * 0.7 + out2 * 0.3).cpu().numpy()

        for i, fname in enumerate(fnames):
            # [3, H, W] → [H, W, 3] → uint8 → flatten to 49152 ints
            img_hwc = (sr[i].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            flat    = img_hwc.reshape(-1).tolist()
            rows.append({'Id': fname, 'Pixels': ' '.join(map(str, flat))})

submission = pd.DataFrame(rows, columns=['Id', 'Pixels'])
submission.to_csv(f'{WORK_DIR}/submission.csv', index=False)

# ── SANITY CHECKS ─────────────────────────────────────────────────────────────
assert len(submission) == len(test_ds), f"Row count mismatch: {len(submission)}"
sample_vals = list(map(int, submission['Pixels'].iloc[0].split()))
assert len(sample_vals) == 128 * 128 * 3, f"Expected 49152 values, got {len(sample_vals)}"
assert all(0 <= v <= 255 for v in sample_vals[:2000]), "Pixel value out of range!"

print(f"\nSaved   : {WORK_DIR}/submission.csv")
print(f"Rows    : {len(submission)}  (expected 495)")
print(f"Sample  : {submission['Pixels'].iloc[0][:80]} ...")
print("All sanity checks PASSED")
