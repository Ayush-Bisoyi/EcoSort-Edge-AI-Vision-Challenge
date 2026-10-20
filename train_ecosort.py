"""Train/export an enhanced compact EcoSort classifier.

This version deliberately demonstrates the requested edge-AI techniques:
depthwise-separable convolutions, ReLU6, dropout, MixUp, stronger training
augmentation, AdamW, and OneCycleLR. The exported ONNX model still expects
ImageNet-normalized NCHW float32 input with shape (batch, 3, 96, 96).

Usage (from the repository root, dataset extracted to data/public/{train,val}):
    python train_ecosort.py --data data/public --epochs 40 --out model.onnx
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms


CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise spatial filter followed by a 1x1 channel-mixing convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU6(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch),
            ConvBNReLU6(in_ch, out_ch, kernel=1),
        )
        self.skip = nn.Identity() if stride == 1 and in_ch == out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return x + y if self.skip is not None else y


class EcoSortEfficientNet(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU6(3, 32, stride=2),       # 96 -> 48
            DepthwiseSeparableBlock(32, 48, 2), # 48 -> 24
            DepthwiseSeparableBlock(48, 48),
            DepthwiseSeparableBlock(48, 64, 2), # 24 -> 12
            DepthwiseSeparableBlock(64, 64),
            DepthwiseSeparableBlock(64, 96, 2), # 12 -> 6
            DepthwiseSeparableBlock(96, 96),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def make_datasets(data_dir: str, full_data: bool = False):
    # Spatial affine augmentation runs in the dataset pipeline. The batch-level
    # flips/colour changes below run after transfer to the selected device.
    train_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomAffine(degrees=20, translate=(0.08, 0.08), scale=(0.75, 1.0)),
        transforms.ToTensor(),
    ])
    clean_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=clean_tf)
    assert train_ds.classes == CLASSES
    assert val_ds.classes == CLASSES
    if full_data:
        # Use the augmentation transform for both public splits only after all
        # architecture/hyperparameters have been selected.
        full_ds = ConcatDataset([train_ds, datasets.ImageFolder(os.path.join(data_dir, "val"), transform=train_tf)])
        return full_ds, None
    return train_ds, val_ds


def device_batch_augment(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Per-sample augmentation performed after the batch reaches the device.

    Inputs are in [0, 1]. Besides flips and brightness/contrast jitter, random
    Gaussian noise and blur are applied so the model does not become brittle to
    sensor noise, soft focus or lighting changes (the public images are clean).
    """
    n = x.size(0)

    def coin(p: float) -> torch.Tensor:
        return (torch.rand((n, 1, 1, 1), device=device) < p)

    x = torch.where(coin(0.5), x.flip(-1), x)
    x = torch.where(coin(0.25), x.flip(-2), x)

    # Random blur (3x3 box filter) on ~25% of the images.
    blurred = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    x = torch.where(coin(0.25), blurred, x)

    # Contrast around each image's own mean, then brightness (wide range).
    contrast = torch.empty((n, 1, 1, 1), device=device).uniform_(0.6, 1.4)
    brightness = torch.empty((n, 1, 1, 1), device=device).uniform_(0.5, 1.5)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * brightness

    # Gaussian noise (sd up to ~0.14, i.e. ~35/255) on ~50% of the images.
    sd = torch.empty((n, 1, 1, 1), device=device).uniform_(0.0, 0.14) * coin(0.5)
    x = x + torch.randn_like(x) * sd

    # Occasional grayscale.
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
    x = torch.where(coin(0.1), gray, x)
    return x.clamp(0.0, 1.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.20):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    good = total = 0
    for x, y in loader:
        out = model(normalize(x.to(device)))
        good += (out.argmax(1).cpu() == y).sum().item()
        total += y.numel()
    return 100.0 * good / total


def export_onnx(model: nn.Module, path: str) -> None:
    model.eval().cpu()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 96, 96),
        path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/public", help="folder containing train/ and val/")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--full-data", action="store_true", help="Train on public train+val after settings are selected")
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_ds, val_ds = make_datasets(args.data, full_data=args.full_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EcoSortEfficientNet(len(CLASSES)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"device={device} train={len(train_ds)} val={0 if val_ds is None else len(val_ds)} params={params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.25, div_factor=10.0, final_div_factor=100.0,
    )
    best_acc = float('nan')
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = good = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = device_batch_augment(x, device)
            x = normalize(x)
            x, ya, yb, lam = mixup(x, y)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = lam * criterion(out, ya) + (1.0 - lam) * criterion(out, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            good += (out.argmax(1) == y).sum().item()
            total += y.numel()
        if val_loader is not None:
            val_acc = accuracy(model, val_loader, device)
            best_acc = val_acc  # final-epoch score; the checkpoint is NOT chosen using val
        else:
            val_acc = float("nan")
        print(f"epoch={epoch:02d} loss={loss_sum/total:.4f} train_mixup_acc={100*good/total:5.1f}% val={val_acc:5.1f}% lr={scheduler.get_last_lr()[0]:.6f}")

    if val_loader is not None:
        print(f"final_epoch_val={best_acc:.2f}%  (exported model = last epoch)")
    else:
        print("full_data_training_complete=true (no independent validation score)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.out)
    print(f"wrote={args.out} bytes={os.path.getsize(args.out):,}")


if __name__ == "__main__":
    main()
