#!/usr/bin/env python3
"""Frozen pretrained vision features for a FAIR vision baseline.

Why this exists
---------------
The first version of the early-warning comparison gave "RGB" a 32K-parameter hand-rolled
CNN on a 64x64 grey down-sample of a single camera. That is not a vision baseline anyone
would accept: it scored below chance (epAUC 0.20) against a 32K tactile encoder. Any
"touch beats vision" claim built on it is worthless.

Here the RGB branch gets a FROZEN pretrained encoder -- ImageNet ResNet-18 or DINOv2
ViT-L/14 (both already cached on this machine) -- and only a small head is trained, which
is the standard way to give a modality its best shot at the same data budget.

Features are cached to disk so repeated runs are cheap.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE = Path("/mnt/public/datasets/bench2dex/vision_feats")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _prep(rgb: np.ndarray, size: int) -> "np.ndarray":
    """(N,H,W,3) uint8 -> (N,3,size,size) float32, ImageNet-normalised."""
    import cv2
    out = np.empty((len(rgb), 3, size, size), np.float32)
    for i, im in enumerate(rgb):
        r = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        v = r.astype(np.float32) / 255.0
        v = (v - IMAGENET_MEAN) / IMAGENET_STD
        out[i] = v.transpose(2, 0, 1)
    return out


def encode(rgb: np.ndarray, encoder: str, dev: str = "cuda", tag: str = "",
           size: int = 224, batch: int = 64) -> np.ndarray:
    """Return frozen features (N, D) float32.

    `rgb` may be (N,H,W,3) for one camera or (N,C,H,W,3) for several; with several the
    per-camera features are concatenated, so a visuo-tactile comparison gives vision the
    WRIST cameras too (the main visual source in manipulation), not just an overhead view.
    """
    if rgb.ndim == 5:
        parts = [encode(rgb[:, c], encoder, dev, tag=f"{tag}_cam{c}", size=size, batch=batch)
                 for c in range(rgb.shape[1])]
        return np.concatenate(parts, axis=1)
    key = hashlib.sha1(f"{encoder}|{size}|{tag}|{len(rgb)}|{rgb.shape}".encode()).hexdigest()[:16]
    path = CACHE / f"{encoder}_{key}.npy"
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            pass

    import torch
    import torch.nn as nn
    import torchvision

    if encoder == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
    elif encoder in ("dinov2", "dinov2_vitl14"):
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
    else:
        raise ValueError(f"unknown encoder {encoder!r}")
    m.eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)

    x = _prep(rgb, size)
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            b = torch.from_numpy(x[i:i + batch]).to(dev)
            f = m(b)
            if isinstance(f, dict):
                f = f.get("x_norm_clstoken", next(iter(f.values())))
            feats.append(f.float().cpu().numpy())
    out = np.concatenate(feats)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, out.astype(np.float16))
    return out
