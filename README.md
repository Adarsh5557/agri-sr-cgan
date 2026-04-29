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
