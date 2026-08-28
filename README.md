# Vision Transformer vs CNN — CIFAR-100

A from-scratch comparison of a Vision Transformer (ViT) and a custom CNN baseline for image classification on CIFAR-100 using PyTorch.

## Overview

This project implements both models from scratch and compares them under a controlled training setup. The ViT uses patch-based image tokenization, multi-head self-attention, Transformer encoder blocks, a learnable class token, and positional embeddings. The CNN baseline uses convolutional blocks, batch normalization, ReLU activations, max pooling, and global average pooling.

## Vision Transformer

- Input: 32×32 RGB images
- Patch size: 4×4
- 64 image patches + 1 `[CLS]` token
- Embedding dimension: 192
- 3 attention heads
- 9 Transformer encoder blocks
- MLP ratio: 4
- Layer normalization and residual connections
- GELU activations and dropout

## CNN Baseline

The custom CNN uses four convolutional stages with increasing channel widths, BatchNorm, ReLU activations, max pooling, and adaptive global average pooling before classification.

## Training

- Dataset: CIFAR-100
- Loss: Cross-Entropy Loss
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 0.05
- Scheduler: Cosine Annealing
- Batch size: 128
- Mixed-precision GPU training
- Data augmentation: random crop and horizontal flip

## Results

The recorded CIFAR-100 evaluation from the original experiment:

| Model | Top-1 Accuracy | Top-5 Accuracy |
|---|---:|---:|
| Vision Transformer | 58.39% | 82.46% |
| CNN Baseline | 71.42% | 91.58% |

The models were also designed with comparable parameter counts (approximately 4.03M for ViT and 4.35M for CNN), enabling a more meaningful architectural comparison.

## Tech Stack

Python, PyTorch, Torchvision, NumPy, Matplotlib, CUDA

## Project Structure

```text
Vision-Transformer-vs-CNN-CIFAR/
├── vision_transformer_cifar.py
├── vision_transformer_cifar.ipynb
├── README.md
└── requirements.txt
```

## Running Locally

```bash
pip install -r requirements.txt
python vision_transformer_cifar.py
```

The CIFAR-100 dataset is downloaded automatically by Torchvision. GPU execution is recommended for training.
